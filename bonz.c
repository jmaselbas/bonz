#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <fcntl.h>
#include <errno.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "glad.h"
#include <SDL.h>

#define SINGLE_WIN 1

#define LEN(a) (sizeof(a)/sizeof(*a))
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#define MIN(a,b) ((a)<(b) ? (a) : (b))

#define GLSL_VERSION "#version 400 core\n"

struct texture {
	GLenum unit;
	GLenum type;
	GLuint id;
};

static SDL_Window *win_live;
static SDL_Window *win_ctrl;
static unsigned int default_width = 1080;
static unsigned int default_height = 800;

static SDL_GLContext gl_ctx;
static double time_start;
static double xpos, ypos;
static int buttons[8];

static inline int
mouse_click(int b)
{
	return buttons[b] == SDL_PRESSED;
}
static inline int mouse_left_click(void) { return mouse_click(1); }
static inline int mouse_middle_click(void) { return mouse_click(2); }
static inline int mouse_right_click(void) { return mouse_click(3); }

static int verbose;
static int show_gui;
static char *argv0;
static GLuint quad_vao;
static GLuint quad_vbo;
static GLuint vshd;

struct shader {
	GLuint prog;
	GLuint fshd;
	char *name;
	time_t time;
};
static size_t shader_count;
static struct shader shaders[16];
static GLuint shaders_fbo;
static struct texture tex_shd;
static struct shader *shader;

static struct texture tex_gui;
static struct texture tex_snd;
static struct texture tex_fft;
static struct texture tex_fft_smth;
static float smth_fac = 0.9;

static char *frag;
static size_t frag_size;

static char logbuf[4096];
static GLsizei logsize;
static unsigned char midi_cc_last[128];
static unsigned char midi_cc[16][128];

#include <fftw3.h>
#define FFT_SIZE 2048
static float fftw_in[FFT_SIZE], fftw_out[FFT_SIZE];
static float fft_smth[FFT_SIZE];
static fftwf_plan plan;

#include <jack/jack.h>
#include <jack/midiport.h>

static jack_client_t *jack;
static jack_port_t *midi_port;
static jack_port_t *input_port;

#include "qoi.h"

#define GUI_IMPLEMENTATION
#include "gui.h"
static struct gui_state gui_state;
static GLuint gui_prg;

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
create_tex(GLenum type)
{
	struct texture tex = {0};
	static GLuint unit = 0;

	tex.unit = unit++;
	tex.type = type;
	glGenTextures(1, &tex.id);

	return tex;
}

static struct texture
create_2drgb_tex(size_t w, size_t h, void *data)
{
	struct texture tex = create_tex(GL_TEXTURE_2D);

	glBindTexture(tex.type, tex.id);

	glTexParameteri(tex.type, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(tex.type, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(tex.type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(tex.type, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(tex.type, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

	return tex;
}

static struct texture
create_1dr32_tex(size_t size, void *data)
{
	struct texture tex = create_tex(GL_TEXTURE_1D);

	glBindTexture(tex.type, tex.id);
	glTexParameteri(tex.type, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(tex.type, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage1D(tex.type, 0, GL_R32F, size, 0, GL_RED, GL_FLOAT, data);

	return tex;
}

static void
update_1dr32_tex(struct texture *tex, void *data, size_t size)
{
	glBindTexture(GL_TEXTURE_1D, tex->id);
	glTexSubImage1D(GL_TEXTURE_1D, 0, 0, size, GL_RED, GL_FLOAT, data);
}

static int
shader_compile(GLuint shd, const GLchar *txt, GLint len)
{
	int ret;

	glShaderSource(shd, 1, &txt, &len);
	glCompileShader(shd);
	glGetShaderiv(shd, GL_COMPILE_STATUS, &ret);
	if (!ret) {
		glGetShaderInfoLog(shd, sizeof(logbuf), &logsize, logbuf);
		fprintf(stderr, "--- ERROR ---\n%s", logbuf);
	}
	return ret;
}

static int
shader_link(GLuint prog, GLuint vert, GLuint frag)
{
	int ret;

	glAttachShader(prog, vert);
	glAttachShader(prog, frag);
	glLinkProgram(prog);
	glGetProgramiv(prog, GL_LINK_STATUS, &ret);
	if (!ret) {
		glGetProgramInfoLog(prog, sizeof(logbuf), &logsize, logbuf);
		printf("--- ERROR ---\n%s", logbuf);
	}
	return ret;
}

static void
shader_reload(struct shader *s)
{
	GLuint nprg;
	GLuint fshd = glCreateShader(GL_FRAGMENT_SHADER);
	FILE *file = fopen(s->name, "r");
	long size = 0;
	const GLchar *src;
	GLint len;
	GLint loc;

	if (!file) {
		fprintf(stderr, "%s: %s\n", s->name, strerror(errno));
		return;
	}

	fseek(file, 0, SEEK_END);
	size = ftell(file);
	if (size < 0) {
		fprintf(stderr, "%s: ftell: %s\n", s->name, strerror(errno));
		return;
	}
	fseek(file, 0, SEEK_SET);
	if ((size_t)size >= frag_size)
		frag = realloc(frag, frag_size = size + 1024);
	fread(frag, sizeof(char), size, file);
	frag[size] = '\0';
	fclose(file);

	src = frag;
	len = size;
	if (!shader_compile(fshd, src, len)) {
		glDeleteShader(fshd);
		return;
	}
	nprg = glCreateProgram();
	if (!shader_link(nprg, vshd, fshd)) {
		glDeleteProgram(nprg);
		glDeleteShader(fshd);
		return;
	}

	if (s->prog)
		glDeleteProgram(s->prog);
	s->prog = nprg;

	glUseProgram(s->prog);
	glBindVertexArray(quad_vao);

	loc = glGetAttribLocation(s->prog, "a_pos");
	if (loc >= 0) {
		glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
		glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(loc);
		glVertexAttribDivisor(loc, 0);
	}
	glBindVertexArray(0);
	printf("--- LOADED --- (%d)\n", nprg);
}

static void
texture_init(void)
{
	tex_fft = create_1dr32_tex(LEN(fftw_out), fftw_out);
	tex_fft_smth = create_1dr32_tex(LEN(fft_smth), fft_smth);
	tex_snd = create_1dr32_tex(LEN(fftw_in), fftw_in);
}

static void
shader_init(void)
{
	static float quad[] = {
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0,
	};
	const char *vert =
		GLSL_VERSION
		"layout (location = 0) in vec2 a_pos;\n"
		"out vec2 texcoord;\n"
		"void main() {\n"
		"	gl_Position = vec4(a_pos - 0.5, 0.0, 0.5);\n"
		"	texcoord = a_pos;\n"
		"}\n";

	glGenVertexArrays(1, &quad_vao);
	glBindVertexArray(quad_vao);

	glGenBuffers(1, &quad_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
	glBindVertexArray(0);

	vshd = glCreateShader(GL_VERTEX_SHADER);
	if (!shader_compile(vshd, vert, strlen(vert)))
		die("error in vertex shader\n");

	glEnable(GL_BLEND);

	tex_shd = create_2drgb_tex(128, 1 + LEN(shaders) * 128, NULL);
	glGenFramebuffers(1, &shaders_fbo);
}

static void
update_cc(GLuint sprg, int loc, unsigned char cc)
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
	size_t i;

	while (SDL_PollEvent(&e)) {
		switch (e.type) {
		case SDL_WINDOWEVENT:
			if (e.window.event == SDL_WINDOWEVENT_CLOSE) {
				fini();
				exit(0);
			}
			break;
		case SDL_QUIT:
			fini();
			exit(0);
			break;
		case SDL_KEYDOWN:
			switch (e.key.keysym.sym) {
			case SDLK_TAB:
				show_gui = !show_gui;
				break;
			case SDLK_v:
				verbose = !verbose;
				printf("--- %s ---\n", verbose ? "verbose" : "quiet");
				break;
			case SDLK_r:
				for (i = 0; i < shader_count; i++)
					shader_reload(&shaders[i]);
				break;
			case SDLK_p:
				panic();
				break;
			case SDLK_1:
			case SDLK_2:
			case SDLK_3:
			case SDLK_4:
			case SDLK_5:
			case SDLK_6:
			case SDLK_7:
			case SDLK_8:
			case SDLK_9:
			case SDLK_0:
				i = e.key.keysym.sym - '0';
				if (i >= shader_count)
					break;
				shader = &shaders[i];
				break;
			}
			break;
		case SDL_KEYUP:
		case SDL_MOUSEMOTION:
			xpos = e.motion.x;
			ypos = e.motion.y;
			break;
		case SDL_MOUSEBUTTONDOWN:
		case SDL_MOUSEBUTTONUP:
			buttons[e.button.button] = e.button.state;
			break;
		}
	}
}

static void
update_shader(struct shader *s)
{
	GLint loc;
	char cc[] = "cc000";
	char ccc[] = "c00cc000";
	int i, j;
	GLuint sprg = s->prog;
	float time = get_time() - time_start;

	loc = glGetUniformLocation(sprg, "fGlobalTime");
	if (loc >= 0)
		glProgramUniform1f(sprg, loc, time);
	loc = glGetUniformLocation(sprg, "time");
	if (loc >= 0)
		glProgramUniform1f(sprg, loc, time);

	for (i = 0; i < 128; i++) {
		snprintf(cc, sizeof(cc), "cc%d", i);
		loc = glGetUniformLocation(sprg, cc);
		if (loc >= 0)
			update_cc(sprg, loc, midi_cc_last[i]);

		for (j = 0; j < 16; j++) {
			snprintf(ccc, sizeof(ccc), "c%dcc%d", j, i);
			loc = glGetUniformLocation(sprg, ccc);
			if (loc >= 0)
				update_cc(sprg, loc, midi_cc[j][i]);
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
render_shader(struct shader *s, int x, int y, int w, int h)
{
	GLint loc = glGetUniformLocation(s->prog, "v2Resolution");
	if (loc >= 0)
		glProgramUniform2f(s->prog, loc, w-x, h-y);

	glBindVertexArray(quad_vao);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

static void
render_window(SDL_Window *window)
{
	int w, h;

	SDL_GL_MakeCurrent(window, gl_ctx);
	SDL_GL_GetDrawableSize(window, &w, &h);
	glViewport(0, 0, w, h);
	glClear(GL_COLOR_BUFFER_BIT);

	if (shader->prog) {
		glUseProgram(shader->prog);
		update_shader(shader);
		render_shader(shader, 0, 0, w, h);
	}
}

static void
init_gui(void)
{
	GLuint nprg = glCreateProgram();
	GLuint vshd = glCreateShader(GL_VERTEX_SHADER);
	GLuint fshd = glCreateShader(GL_FRAGMENT_SHADER);
	const char *vert =
		GLSL_VERSION
		"layout (location = 0) in vec2 a_pos;\n"
		"layout (location = 1) in vec4 a_pos_xfrm;\n"
		"layout (location = 2) in vec4 a_shp_xfrm;\n"
		"layout (location = 3) in vec4 a_col_xfrm;\n"
		"out vec2 v_shape;\n"
		"out vec2 v_color;\n"
		"vec2 xfrm(vec4 x) { return a_pos * x.zw + x.xy; }\n"
		"void main() {\n"
		"	gl_Position = vec4(xfrm(a_pos_xfrm), 0.0, 0.5);\n"
		"	v_shape = xfrm(a_shp_xfrm);\n"
		"	v_color = xfrm(a_col_xfrm);\n"
		"}\n";
	const char *frag =
		GLSL_VERSION
		"in vec2 v_shape;\n"
		"in vec2 v_color;\n"
		"out vec4 out_color;\n"
		"uniform sampler2D t_shape;"
		"uniform sampler2D t_color;"
		"void main() {\n"
		"	if (texture(t_shape, v_shape).r < 0.5) discard;\n"
		"	out_color = vec4(texture(t_color, v_color).rgb, 1.0);\n"
		"}\n";
	const char *file = "ascii.qoi";
	qoi_desc desc;
	void *data = qoi_read(file, &desc, 3);

	if (!data)
		die("%s: qoi_read: %s\n", file, strerror(errno));
	/* hack: set the first pixel to 0xffffff */
	memcpy(data, (unsigned char[3]){255,255,255}, 3 * sizeof(char));
	tex_gui = create_2drgb_tex(desc.width, desc.height, data);
	free(data);

	if (!shader_compile(vshd, vert, strlen(vert)))
		die("gui: error in vertex shader\n");
	if (!shader_compile(fshd, frag, strlen(frag)))
		die("gui: error in fragment shader\n");
	if (!shader_link(nprg, vshd, fshd))
		die("gui: error in program link\n");

	gui_prg = nprg;
	gui_init(&gui_state, gui_prg);
}

/* font size */
static float FW = 7.0;
static float FH = 9.0;

void
gui_draw(int w, int h, GLuint prog, GLuint tex_s,  GLuint tex_c)
{
	struct gui_cmd *cmd;
	struct gui_rect r, clip = { 0, 0, w, h };
	struct gui_quad q;
	GLint utex;

	if (gui->cmd_queue_size == 0)
		return;

	glUseProgram(prog);

	utex = glGetUniformLocation(prog, "t_shape");
	if (utex >= 0) {
		glActiveTexture(GL_TEXTURE0 + 1);
		glUniform1i(utex, 1);
		glBindTexture(GL_TEXTURE_2D, tex_s);
	}

	utex = glGetUniformLocation(prog, "t_color");
	if (utex >= 0) {
		glActiveTexture(GL_TEXTURE0 + 2);
		glUniform1i(utex, 2);
		glBindTexture(GL_TEXTURE_2D, tex_c);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 128, 1, GL_RGB, GL_UNSIGNED_BYTE, gui->colors);
	}

	glBindVertexArray(gui->vao);

	gui_for_each_cmd(cmd) {
		size_t i;
		float ox;
		char c;
		switch (cmd->type) {
		case GUI_TEXT:
			ox = cmd->text.x;
			q.img_xfrm[0] = cmd->text.col/128.0;
			q.img_xfrm[1] = 0;
			q.img_xfrm[2] = 0;
			q.img_xfrm[3] = 0;
			for (i = 0; i < cmd->text.len; i++, ox += FW) {
				c = cmd->text.str[i];
				q.pos_xfrm[0] = -0.5 + ox/(float)w;
				q.pos_xfrm[1] = +0.5 - cmd->text.y/(float)h;
				q.pos_xfrm[2] = +(FW)/(float)w;
				q.pos_xfrm[3] = -(FH)/(float)h;

				if (c <= ' ') continue;
				q.shp_xfrm[0] = ((int)(c - ' ') % 16) / 16.0;
				q.shp_xfrm[1] = ((int)(c - ' ') / 16) / 6.0;
				q.shp_xfrm[2] = 1.0/16.0;
				q.shp_xfrm[3] = 1.0/6.0;

				r.x = ox;
				r.y = cmd->text.y;
				r.w = FW;
				r.h = FH;
				if (gui_rect_overlap(r, clip))
					gui_push_quad(q);
			}
			break;
		case GUI_SHAPE:
			q.pos_xfrm[0] = -0.5 + cmd->shape.rect.x/(float)w;
			q.pos_xfrm[1] = +0.5 - cmd->shape.rect.y/(float)h;
			q.pos_xfrm[2] = +((cmd->shape.rect.w)/(float)w);
			q.pos_xfrm[3] = -((cmd->shape.rect.h)/(float)h);

			q.shp_xfrm[0] = 0;
			q.shp_xfrm[1] = 0;
			q.shp_xfrm[2] = 0;
			q.shp_xfrm[3] = 0;
			{
			float ww = 128.0;
			float hh = 1 + 16 * 128.0;
			q.img_xfrm[0] = cmd->shape.image.x/ww;
			q.img_xfrm[1] = (cmd->shape.image.y+cmd->shape.image.h)/hh;
			q.img_xfrm[2] = cmd->shape.image.w/ww;
			q.img_xfrm[3] = -cmd->shape.image.h/hh;
			}
			if (gui_rect_overlap(cmd->shape.rect, clip))
				gui_push_quad(q);

			break;
		}
	}
	gui_flush_draw_queue();
	glBindVertexArray(0);
}

static void
gui_draw_grid_elem(size_t idx, int px, int py, size_t sz)
{
	uint8_t col;
	if (idx < shader_count)
		col = gui_color(40, 40, 40);
	else
		col = gui_color(20, 20, 20);
	if (&shaders[idx] == shader) {
		gui_fill(px-4, py-4, sz+8, sz+8, gui_color(255, 0, 0));
	} else if (idx < shader_count && gui_mouse_in((struct gui_rect){px, py, sz, sz})) {
		gui_fill(px-4, py-4, sz+8, sz+8, gui_color(80, 80, 80));
		col = gui_color(80, 80, 80);
		if (mouse_left_click())
			shader = &shaders[idx];
	}

	if (idx < shader_count)
		gui_image(gui_rect(px, py, sz, sz),
			  gui_rect(0, 1 + idx * 128, 128, 128));
	else
		gui_fill(px, py, sz, sz, col);
	if (idx < shader_count) {
		char buf[] = "123467890";
		snprintf(buf, sizeof(buf), "%zd", idx);
		gui_fill(px, py, FW * strlen(buf), FH, col);
		gui_text(px, py, buf, gui_color(255, 255, 255));

		gui_fill(px, py+sz, FW * strlen(shaders[idx].name), FH, col);
		gui_text(px, py+sz, shaders[idx].name, gui_color(255, 255, 255));
	}
}

static void
gui_view_grid(void)
{
	size_t size, i;
	int ix, iy;
	int px, py;
	int padx, pady;
	int w, h;
	SDL_GL_GetDrawableSize(win_ctrl, &w, &h);

	size = MIN(w / (2*4+5), h / (2*4+5));

	padx = (w - (2*4+5) * size) / 2;
	pady = (h - (2*4+5) * size) / 2;
	for (i = 0; i < LEN(shaders); i++) {
		ix = i % 4;
		iy = i / 4;
		px = ix*(3 * size) + size + padx;
		py = iy*(3 * size) + size + pady;
		gui_draw_grid_elem(i, px, py, 2*size);
	}
}

static void
render(void)
{
	int w, h;

#ifndef SINGLE_WIN
	render_window(win_live);
	SDL_GL_SwapWindow(win_live);
#endif

	glBindFramebuffer(GL_FRAMEBUFFER, shaders_fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex_shd.id, 0);
	for (size_t i = 0; i < shader_count; i++) {
		glUseProgram(shaders[i].prog);
		update_shader(&shaders[i]);
		render_shader(&shaders[i], 0, 1 + i * 128, 128, 128);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	render_window(win_ctrl);

	if (show_gui) {
		gui_begin(&gui_state);
		gui_view_grid();
		SDL_GL_GetDrawableSize(win_ctrl, &w, &h);
		gui_draw(w, h, gui_prg, tex_gui.id, tex_shd.id);
	}
	SDL_GL_SwapWindow(win_ctrl);
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
	if (jack) {
		jack_deactivate(jack);
		jack_client_close(jack);
		jack = NULL;
	}
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
	jack_options_t options = JackNoStartServer;
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

static void
shader_poll(struct shader *s)
{
	struct stat sb;
	int ret;

	ret = stat(s->name, &sb);
	if (ret < 0) {
		/* file is probably beeing saved */
		if (errno != ENOENT)
			fprintf(stderr, "stat '%s': %s\n", s->name, strerror(errno));
		return;
	}

	if (s->time != sb.st_ctime) {
		s->time = sb.st_ctime;
		shader_reload(s);
	}
}

static void
sdl_gl_init(void)
{
	SDL_Window *window;

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
				  default_width, default_height,
				  SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

	if (!window)
		die("Failed to create window: %s\n", SDL_GetError());

	gl_ctx = SDL_GL_CreateContext(window);
	if (!gl_ctx)
		die("Failed to create openGL context: %s\n", SDL_GetError());

	if (!gladLoadGLLoader((GLADloadproc) SDL_GL_GetProcAddress))
		die("GL init failed\n");

	win_live = window;
#if SINGLE_WIN
	win_ctrl = win_live;
#else
	win_ctrl = SDL_CreateWindow("ctrl", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
				    default_width, default_height,
				    SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
#endif
}

static void
init(void)
{
	sdl_gl_init();
	time_start = get_time();

	jack_init();
	shader_init();
	texture_init();
	init_gui();
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

static int
is_file(const char *file)
{
	struct stat stat;
	int fd, ret;

	fd = open(file, O_RDONLY);
	if (fd == -1) {
		fprintf(stderr, "open %s: %s\n", file, strerror(errno));
		return 0;
	}
	ret = fstat(fd, &stat);
	close(fd);
	if (ret == -1) {
		fprintf(stderr, "stat %s: %s\n", file, strerror(errno));
		return 0;
	}

	return S_ISREG(stat.st_mode);
}

int
main(int argc, char **argv)
{
	size_t i;

	argv0 = argv[0];

	if (argc < 2)
		usage();

	for (i = 1; (int)i < argc && shader_count < LEN(shaders); i++) {
		if (!is_file(argv[i]))
			die("%s: is not a regular file\n", argv[i]);
		shaders[shader_count++].name = argv[i];
	}
	shader = &shaders[0];

	plan = fftwf_plan_r2r_1d(FFT_SIZE, fftw_in, fftw_out, FFTW_REDFT10, FFTW_MEASURE);

	init();
	while (1) {
		input();
		for (i = 0; i < shader_count; i++)
			shader_poll(&shaders[i]);
		render();
	}
	fini();

	return 0;
}
