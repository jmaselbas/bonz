#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <fcntl.h>
#include <errno.h>

#include <sys/stat.h>

#include "glad.h"
#include <SDL.h>

struct texture {
	GLenum unit;
	GLenum type;
	GLuint id;
};

#define LEN(a) (sizeof(a)/sizeof(*a))

SDL_Window *window;
SDL_GLContext context;
double time_start;

int verbose;
char *argv0;
unsigned int width = 1080;
unsigned int height = 800;
GLuint quad_vao;
GLuint quad_vbo;
GLuint sprg;
GLuint vshd;
GLuint fshd;

struct texture tex_snd;
struct texture tex_fft;
struct texture tex_fft_smth;
float smth_fac = 0.9;

char *frag_name;
GLchar *frag;
GLuint frag_size;

char logbuf[4096];
GLsizei logsize;
unsigned char midi_cc_last[128];
unsigned char midi_cc[16][128];

#include <fftw3.h>
#define FFT_SIZE 2048
float fftw_in[FFT_SIZE], fftw_out[FFT_SIZE];
float fft_smth[FFT_SIZE];
fftwf_plan plan;

#include <jack/jack.h>
#include <jack/midiport.h>

jack_client_t *jack;
jack_port_t *midi_port;
jack_port_t *input_port;

static void fini(void);

static void
die(const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	vfprintf(stderr, fmt, ap);
	va_end(ap);

	fini();
	exit(1);
}

#define MSEC_PER_SEC 1000
static double
get_time(void)
{
	return SDL_GetTicks() / (double) MSEC_PER_SEC;
}

static void
panic(void)
{
	int i;
	if (verbose)
		printf("panic\n");
	time_start = get_time();
	memset(midi_cc_last, 0, sizeof(midi_cc_last));
	for (i = 0; i < 16; i++) {
		memset(midi_cc[i], 0, sizeof(midi_cc_last));
	}
}

static struct texture
create_1dr32_tex(void *data, size_t size)
{
	struct texture tex;
	static GLuint unit = 0;

	tex.unit = unit++;
	tex.type = GL_TEXTURE_1D;

	glGenTextures(1, &tex.id);
	glBindTexture(GL_TEXTURE_1D, tex.id);

	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, size, 0, GL_RED, GL_FLOAT, data);

	return tex;
}

static void
update_1dr32_tex(struct texture *tex, void *data, size_t size)
{
	glBindTexture(GL_TEXTURE_1D, tex->id);
	glTexSubImage1D(GL_TEXTURE_1D, 0, 0, size, GL_RED, GL_FLOAT, data);
}

static void
shader_reload(void)
{
	GLuint nprg = glCreateProgram();
	GLuint fshd = glCreateShader(GL_FRAGMENT_SHADER);
	static GLuint pfrag;
	FILE *frag_file = fopen(frag_name, "r");
	size_t size = 0;
	int ret;

	if (!frag_file) {
		fprintf(stderr, "%s: %s\n", frag_name, strerror(errno));
		return;
	}

	while (!feof(frag_file)) {
		if ((size + 1024) > frag_size)
			frag = realloc(frag, 1 + (frag_size += 1024));
		size += fread(frag + size, sizeof(char), 1024, frag_file);
	}
	frag[size] = '\0';
	fclose(frag_file);

	glShaderSource(fshd, 1, (const GLchar * const*)&frag, &frag_size);
	glCompileShader(fshd);
	glGetShaderiv(fshd, GL_COMPILE_STATUS, &ret);
	if (!ret) {
		glGetShaderInfoLog(fshd, sizeof(logbuf), &logsize, logbuf);
		glDeleteProgram(nprg);
		glDeleteShader(fshd);
		printf("--- ERROR ---\n%s", logbuf);
		return;
	}

	glAttachShader(nprg, vshd);
	glAttachShader(nprg, fshd);
	glLinkProgram(nprg);
	glGetProgramiv(nprg, GL_LINK_STATUS, &ret);
	if (!ret) {
		glGetProgramInfoLog(nprg, sizeof(logbuf), &logsize, logbuf);
		glDeleteProgram(nprg);
		glDeleteShader(fshd);
		printf("--- ERROR ---\n%s", logbuf);
		return;
	}

	if (!sprg) {
		sprg = nprg;
	} else {
		glDetachShader(sprg, pfrag);
		glAttachShader(sprg, fshd);
		glLinkProgram(sprg);
		glGetProgramiv(sprg, GL_LINK_STATUS, &ret);
		if (!ret) {
			glGetProgramInfoLog(sprg, sizeof(logbuf), &logsize, logbuf);
			printf("--- ERROR ---\n%s", logbuf);
			return;
		}

		glDeleteProgram(nprg);
	}
	printf("--- LOADED --- (%d) %d\n", nprg, sprg);
	pfrag = fshd;
	glUseProgram(sprg);
}

static void
shader_init(void)
{
	static float quad[] = {
		-1.0, -1.0,  0.5, 0.0, 0.0,
		-1.0,  1.0,  0.5, 0.0, 1.0,
		 1.0, -1.0,  0.5, 1.0, 0.0,
		 1.0,  1.0,  0.5, 1.0, 1.0,
	};
	const char *vert =
		"#version 410 core\n"
		"in vec3 in_pos;\n"
		"in vec2 in_texcoord;\n"
		"out vec2 texcoord;\n"
		"void main()\n"
		"{\n"
		"  gl_Position = vec4(in_pos.x, in_pos.y, in_pos.z, 1.0);\n"
		"  texcoord = in_texcoord;\n"
		"}\n";
	GLint size = strlen(vert);
	int ret;

	tex_fft = create_1dr32_tex(fftw_out, LEN(fftw_out));
	tex_fft_smth = create_1dr32_tex(fft_smth, LEN(fft_smth));
	tex_snd = create_1dr32_tex(fftw_in, LEN(fftw_in));

	glGenVertexArrays(1, &quad_vao);
	glBindVertexArray(quad_vao);

	glGenBuffers(1, &quad_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
	glBindVertexArray(0);

	vshd = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vshd, 1, &vert, &size);
	glCompileShader(vshd);
	glGetShaderInfoLog(vshd, sizeof(logbuf), &logsize, logbuf);
	glGetShaderiv(vshd, GL_COMPILE_STATUS, &ret);
	if (!ret) {
		glDeleteShader(fshd);
		printf("--- ERROR ---\n%s", logbuf);
		die("error in vertex shader\n");
	}
}

static void
render(void)
{
	GLint position;
	GLint texcoord;
	size_t stride;
	size_t offset;

	if (!sprg)
		return;

	glUseProgram(sprg);
	glBindVertexArray(quad_vao);

	stride = 5 * sizeof(float);
	offset = 3 * sizeof(float);

	position = glGetAttribLocation(sprg, "in_pos");
	if (position >= 0) {
		glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, stride, NULL);
		glEnableVertexAttribArray(position);
	}

	texcoord = glGetAttribLocation(sprg, "in_texcoord");
	if (texcoord >= 0) {
		glVertexAttribPointer(texcoord, 2, GL_FLOAT, GL_FALSE, stride, (GLvoid *)offset);
		glEnableVertexAttribArray(texcoord);
	}

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

static void
update_cc(int loc, unsigned char cc)
{
	GLenum type;

	glGetActiveUniform(sprg, loc, 0, NULL, NULL, &type, NULL);

	if (type == GL_FLOAT)
		glProgramUniform1f(sprg, loc, cc / 127.0f);
	else
		glProgramUniform1ui(sprg, loc, cc);
}

static void
input(void)
{
	SDL_Event e;
	while (SDL_PollEvent(&e)) {
		switch(e.type) {
		case SDL_QUIT:
			fini();
			exit(0);
			break;
		case SDL_KEYDOWN:
			switch (e.key.keysym.sym) {
			case SDLK_v:
				verbose = !verbose;
				printf("--- %s ---\n", verbose ? "verbose" : "quiet");
				break;
			case SDLK_r:
				shader_reload();
				break;
			case SDLK_p:
				panic();
				break;
			}
			break;
		case SDL_KEYUP:
		case SDL_MOUSEMOTION:
		case SDL_MOUSEBUTTONDOWN:
		case SDL_MOUSEBUTTONUP:
		case SDL_WINDOWEVENT:
			break;
		}
	}
}

static void
update(void)
{
	GLint loc;
	char cc[] = "cc000";
	char ccc[] = "c00cc000";
	int i, j;

	loc = glGetUniformLocation(sprg, "fGlobalTime");
	if (loc >= 0)
		glProgramUniform1f(sprg, loc, get_time() - time_start);
	loc = glGetUniformLocation(sprg, "time");
	if (loc >= 0)
		glProgramUniform1f(sprg, loc, get_time() - time_start);

	loc = glGetUniformLocation(sprg, "v2Resolution");
	if (loc >= 0)
		glProgramUniform2f(sprg, loc, width, height);

	for (i = 0; i < 128; i++) {
		snprintf(cc, sizeof(cc), "cc%d", i);
		loc = glGetUniformLocation(sprg, cc);
		if (loc >= 0)
			update_cc(loc, midi_cc_last[i]);

		for (j = 0; j < 16; j++) {
			snprintf(ccc, sizeof(ccc), "c%dcc%d", j, i);
			loc = glGetUniformLocation(sprg, ccc);
			if (loc >= 0)
				update_cc(loc, midi_cc[j][i]);
		}
	}

	loc = glGetUniformLocation(sprg, "texFFT");
	if (loc >= 0) {
		glActiveTexture(GL_TEXTURE0 + tex_fft.unit);
		update_1dr32_tex(&tex_fft, fftw_out, LEN(fftw_out));
		glProgramUniform1i(sprg, loc, tex_fft.unit);
	}

	loc = glGetUniformLocation(sprg, "texFFTSmoothed");
	if (loc >= 0) {
		glActiveTexture(GL_TEXTURE0 + tex_fft_smth.unit);
		update_1dr32_tex(&tex_fft_smth, fft_smth, LEN(fft_smth));
		glProgramUniform1i(sprg, loc, tex_fft_smth.unit);
	}

	loc = glGetUniformLocation(sprg, "texSND");
	if (loc >= 0) {
		glActiveTexture(GL_TEXTURE0 + tex_snd.unit);
		update_1dr32_tex(&tex_snd, fftw_in, LEN(fftw_in));
		glProgramUniform1i(sprg, loc, tex_snd.unit);
	}
}

static void
midi_process(size_t size, unsigned char *buff)
{
	unsigned char sts, ccc, ccn, ccv;

	if (size < 2)
		return;

	sts = (buff[0] & 0xf0) >> 4;

	if (sts == 0xb) {
		/* control change */
		ccc = buff[0] % 16;
		ccn = buff[1] % 128;
		ccv = buff[2];
		midi_cc[ccc][ccn] = ccv;
		midi_cc_last[ccn] = ccv;
		if (verbose)
			printf("c%dcc%d = %d\n", ccc, ccn, ccv);
	} else if (sts == 0xf) {
		if (buff[0] == 0xff)
			panic();
		printf("midi sys %x %x\n", buff[0], buff[1]);
	} else {
		printf("midi#%d %x %x\n", buff[0] & 0xf, buff[0] >> 4, buff[1]);
	}
}

#define mix(x,y,a) ((x) * (1 - (a)) + (y) * (a))

static int
jack_process(jack_nframes_t frames, void *arg)
{
	void *buffer;
	jack_nframes_t n, i;
	jack_midi_event_t event;
	jack_default_audio_sample_t *in;
	size_t size = sizeof(jack_default_audio_sample_t);
	int r;

	(void) arg; /* unused */

	if (midi_port) {
		buffer = jack_port_get_buffer(midi_port, frames);

		n = jack_midi_get_event_count(buffer);
		for (i = 0; i < n; i++) {
			r = jack_midi_event_get(&event, buffer, i);
			if (r == 0)
				midi_process(event.size, event.buffer);
		}
	}

	if (input_port) {
		in = jack_port_get_buffer(input_port, frames);
		if (frames < FFT_SIZE) {
			/* shift previous frames */
			memmove(fftw_in, fftw_in + frames, (FFT_SIZE - frames) * size);
		} else {
			/* discard extra frames */
			in += frames - FFT_SIZE;
			frames = FFT_SIZE;
		}
		memcpy(fftw_in + FFT_SIZE - frames, in, frames * size);
		if (plan)
			fftwf_execute(plan);
		for (i = 0; i < LEN(fft_smth); i++)
			fft_smth[i] =  mix(fftw_out[i], fft_smth[i], smth_fac);
	}

	return 0;
}

static void
jack_fini(void)
{
	jack_deactivate(jack);
	jack_client_close(jack);
	jack = NULL;
}

static void
jack_shutdown(void *arg)
{
	(void) arg; /* unused */
	jack_fini();
}

static void
jack_init(void)
{
	jack_options_t options = JackNoStartServer | JackServerName;
	const char **ports;
	int ret;

	jack = jack_client_open(argv0, options, NULL);
	if (!jack) {
		fprintf(stderr, "error connecting jack client\n");
		return;
	}

	jack_on_shutdown(jack, jack_shutdown, NULL);

	ret = jack_set_process_callback(jack, jack_process, 0);
	if (ret) {
		fprintf(stderr, "Could not register process callback.\n");
		return;

	}
	midi_port = jack_port_register(jack, "input", JACK_DEFAULT_MIDI_TYPE, JackPortIsInput, 0);
	if (!midi_port) {
		fprintf(stderr, "Could not register midi port.\n");
		return;
	}

	input_port = jack_port_register(jack, "mono", JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
	if (!input_port) {
		fprintf(stderr, "Could not register audio port.\n");
		return;
	}

	ret = jack_activate(jack);
	if (ret) {
		fprintf(stderr, "Could not activate client.\n");
		return;
	}

	ports = jack_get_ports(jack, NULL, NULL, JackPortIsOutput);
	if (ports) {
		jack_connect(jack, ports[0], jack_port_name(input_port));
	}
}

#include <sys/types.h>
#include <sys/stat.h>

static void
shader_poll(void)
{
	static __time_t last_time;
	struct stat sb;
	int ret;

	ret = stat(frag_name, &sb);
	if (ret < 0) {
		/* file is probably beeing saved */
		if (errno != ENOENT)
			fprintf(stderr, "stat '%s': %s\n", frag_name, strerror(errno));
		return;
	}

	if (last_time != sb.st_ctime) {
		last_time = sb.st_ctime;
		shader_reload();
	}
}

static void
init(void)
{
	if (SDL_InitSubSystem(SDL_INIT_VIDEO))
		die("SDL init failed: %s\n", SDL_GetError());

	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, SDL_TRUE);
	SDL_GL_SetSwapInterval(1);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);

	window = SDL_CreateWindow(argv0, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
				  width, height,
				  SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
				  | SDL_WINDOW_INPUT_FOCUS
				  | SDL_WINDOW_MOUSE_FOCUS
				  | SDL_WINDOW_MAXIMIZED
				  );

	if (!window)
		die("Failed to create window: %s\n", SDL_GetError());

	context = SDL_GL_CreateContext(window);
	if (!context)
		die("Failed to create openGL context: %s\n", SDL_GetError());

	if (!gladLoadGLLoader((GLADloadproc) SDL_GL_GetProcAddress))
		die("GL init failed\n");

	glViewport(0, 0, width, height);

	time_start = get_time();

	jack_init();
	shader_init();
}

static void
fini(void)
{
	jack_fini();
}

static void
usage(void)
{
	printf("usage: %s <shader_file>\n", argv0);
	exit(1);
}

int
main(int argc, char **argv)
{
	struct stat stat;
	int fd, ret;

	argv0 = argv[0];

	if (argc != 2)
		usage();

	fd = open(argv[1], O_RDONLY);
	if (fd == -1)
		die("%s: %s\n", argv[1], strerror(errno));
	ret = fstat(fd, &stat);
	if (ret == -1)
		die("stat: %s\n", argv[1], strerror(errno));
	if (!S_ISREG(stat.st_mode))
		die("%s: is not a regular file\n", argv[1]);
	close(fd);
	frag_name = argv[1];

	plan = fftwf_plan_r2r_1d(FFT_SIZE, fftw_in, fftw_out, FFTW_REDFT10, FFTW_MEASURE);

	init();
	while (1) {
		int w, h;
		SDL_GL_GetDrawableSize(window, &w, &h);
		width  = (w < 0) ? 0 : w;
		height = (h < 0) ? 0 : h;
		glViewport(0, 0, width, height);
		input();
		shader_poll();
		update();
		render();
		SDL_GL_SwapWindow(window);
	}
	fini();

	return 0;
}
