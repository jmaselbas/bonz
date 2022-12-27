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

#undef offsetof
#define offsetof(type, memb) __builtin_offsetof(type, memb)

#define LEN(a) (sizeof(a)/sizeof(*a))
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#define MIN(a,b) ((a)<(b) ? (a) : (b))

#define GLSL_VERSION "#version 400 core\n"

struct texture {
	GLenum unit;
	GLenum type;
	GLuint id;
};

SDL_Window *win_live;
SDL_Window *win_ctrl;
unsigned int default_width = 1080;
unsigned int default_height = 800;

SDL_GLContext gl_ctx;
double time_start;
double xpos, ypos;
int buttons[8];

static inline int
mouse_click(int b)
{
	return buttons[b] == SDL_PRESSED;
}
static inline int mouse_left_click(void) { return mouse_click(1); }
static inline int mouse_middle_click(void) { return mouse_click(2); }
static inline int mouse_right_click(void) { return mouse_click(3); }

float fps[64];
size_t fps_count;
int verbose;
char *argv0;
GLuint quad_vao;
GLuint quad_vbo;
GLuint vshd;

struct shader {
	GLuint prog;
	GLuint fshd;
	char *name;
	time_t time;
};
size_t shader_count;
struct shader shaders[16];
struct shader *shader;

struct texture tex_gui;
struct texture tex_snd;
struct texture tex_fft;
struct texture tex_fft_smth;
float smth_fac = 0.9;

char *frag;
size_t frag_size;

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

#include "qoi.h"

static void gui_text(int x, int y, const char *s);

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
/* linear min/mag filter do not mix well with sampling specific texel */
//	glTexParameteri(tex.type, GL_TEXTURE_WRAP_S, GL_REPEAT);
//	glTexParameteri(tex.type, GL_TEXTURE_WRAP_T, GL_REPEAT);
//	glTexParameteri(tex.type, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//	glTexParameteri(tex.type, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

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

static void
shader_bind_quad(struct shader *s)
{
	GLint position;

	glUseProgram(s->prog);
	glBindVertexArray(quad_vao);

	position = glGetAttribLocation(s->prog, "a_pos");
	if (position >= 0) {
		glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		glEnableVertexAttribArray(position);
		glVertexAttribDivisor(position, 0);
	}
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

	printf("--- LOADED --- (%d)\n", nprg);
	shader_bind_quad(s);
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
		"in vec2 a_pos;\n"
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

struct gui_quad {
	float xfrm[4];
	float tfrm[4];
	float rgba[4];
};

struct gui_cmd {
	enum gui_cmd_type {
		GUI_CLIP,
		GUI_RECT,
		GUI_TEXT,
		GUI_COLOR,
	} type;
	union {
		struct {
			uint8_t r,g,b,a;
		} color;
		struct gui_rect {
			int16_t x, y;
			uint16_t w, h;
		} rect, clip;
		struct {
			int16_t x, y;
			uint8_t len;
			char str[];
		} text;
	};
};
static struct gui_cmd gui_cmd_queue[4096];
static size_t gui_cmd_queue_size;
static GLuint gui_prg;
static GLuint gui_vao;
static GLuint gui_vbo;
static const size_t gui_nmax = 1024;
static struct gui_quad gui_data[1024];
static GLuint gui_count;
static GLuint gui_total_count;
static GLuint gui_draw_count;

static void
gui_begin(void)
{
	gui_cmd_queue_size = 0;
	gui_total_count = 0;
	gui_draw_count = 0;
}

static size_t
gui_cmd_size(struct gui_cmd *cmd)
{
	size_t s = sizeof(*cmd);

	if (cmd->type == GUI_TEXT)
		s += cmd->text.len;

	return s;
}

#define gui_cmd_queue_end() (((void *)gui_cmd_queue) + gui_cmd_queue_size)

static struct gui_cmd *
gui_cmd_next(struct gui_cmd *cmd)
{
	if (cmd)
		cmd = ((void *)cmd) + gui_cmd_size(cmd);
	if (((void *)cmd) >= gui_cmd_queue_end())
		return NULL;
	return cmd;
}

#define gui_for_each_cmd(c) for ((c) = gui_cmd_queue; (c); (c) = gui_cmd_next(c))

static void
gui_text(int x, int y, const char *s)
{
	struct gui_cmd *cmd = gui_cmd_queue_end();
	size_t tlen = strlen(s);
	while (tlen > 0) {
		size_t len = tlen > UINT8_MAX ? UINT8_MAX : tlen;
		size_t size = len + sizeof(struct gui_cmd);
		if (gui_cmd_queue_size + size > sizeof(gui_cmd_queue)) {
			printf("!!\n");
			return;
		}
		cmd->type = GUI_TEXT;
		cmd->text.x = x;
		cmd->text.y = y;
		cmd->text.len = len;
		memcpy(cmd->text.str, s, len);

		tlen -= len;
		s += len;

		gui_cmd_queue_size += size;
	}
}

static void
gui_rect(int x, int y, unsigned int w, unsigned int h)
{
	struct gui_cmd *cmd = gui_cmd_queue_end();
	size_t size = sizeof(struct gui_cmd);
	if (gui_cmd_queue_size + size > sizeof(gui_cmd_queue))
		return;
	cmd->type = GUI_RECT;
	cmd->rect.x = x;
	cmd->rect.y = y;
	cmd->rect.w = w;
	cmd->rect.h = h;

	gui_cmd_queue_size += size;
}

static void
gui_clip(int x, int y, unsigned int w, unsigned int h)
{
	struct gui_cmd *cmd = gui_cmd_queue_end();
	size_t size = sizeof(struct gui_cmd);
	if (gui_cmd_queue_size + size > sizeof(gui_cmd_queue))
		return;
	cmd->type = GUI_CLIP;
	cmd->rect.x = x;
	cmd->rect.y = y;
	cmd->rect.w = w;
	cmd->rect.h = h;

	gui_cmd_queue_size += size;
}

static void
gui_color(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
	struct gui_cmd *cmd = gui_cmd_queue_end();
	size_t size = sizeof(struct gui_cmd);
	if (gui_cmd_queue_size + size > sizeof(gui_cmd_queue))
		return;
	cmd->type = GUI_COLOR;
	cmd->color.r = r;
	cmd->color.g = g;
	cmd->color.b = b;
	cmd->color.a = a;

	gui_cmd_queue_size += size;
}

static void
gui_init(void)
{
	GLuint nprg = glCreateProgram();
	GLuint vshd = glCreateShader(GL_VERTEX_SHADER);
	GLuint fshd = glCreateShader(GL_FRAGMENT_SHADER);
	const char *vert =
		GLSL_VERSION
		"layout (location = 0) in vec2 a_pos;\n"
		"layout (location = 1) in vec4 a_xfrm;\n"
		"layout (location = 2) in vec4 a_tfrm;\n"
		"layout (location = 3) in vec4 a_rgba;\n"
		"out vec2 texcoord;\n"
		"out vec4 color;\n"
		"void main() {\n"
		"	gl_Position = vec4(a_pos * a_xfrm.xy + a_xfrm.zw, 0.0, 0.5);\n"
		"	texcoord = a_pos * a_tfrm.xy + a_tfrm.zw;\n"
		"	color = a_rgba;\n"
		"}\n";
	const char *frag =
		GLSL_VERSION
		"in vec2 texcoord;\n"
		"in vec4 color;\n"
		"out vec4 out_color;\n"
		"uniform sampler2D t_gui;"
		"void main() {\n"
		"	if (texture(t_gui, texcoord).r < 0.5) discard;\n"
		"	out_color = color;\n"
//		"ivec2 p= ivec2(texcoord * textureSize(t_gui, 0));\n"
//		"	out_color = vec4(texelFetch(t_gui, p, 0).rgb, 1.0);\n"
//		"	out_color = vec4(texcoord, 0.0, 1.0);\n"
//		"	out_color = vec4(texture(t_gui, texcoord).rgb, 1.0);\n"
		"}\n";
	const char *file = "ascii.qoi";
	GLuint loc;
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

	glGenVertexArrays(1, &gui_vao);
	glBindVertexArray(gui_vao);

	glGenBuffers(1, &gui_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, gui_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(gui_data), gui_data, GL_STREAM_DRAW);

	loc = 0; /* a_pos */
	glBindBuffer(GL_ARRAY_BUFFER, quad_vbo);
	glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribDivisor(loc, 0);
	glEnableVertexAttribArray(loc);

	loc = 1; /* a_xfrm */
	glBindBuffer(GL_ARRAY_BUFFER, gui_vbo);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE,
			      sizeof(struct gui_quad),
			      (void *)offsetof(struct gui_quad, xfrm));
	glVertexAttribDivisor(loc, 1);
	glEnableVertexAttribArray(loc);

	loc = 2; /* a_tfrm */
	glBindBuffer(GL_ARRAY_BUFFER, gui_vbo);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE,
			      sizeof(struct gui_quad),
			      (void *)offsetof(struct gui_quad, tfrm));
	glVertexAttribDivisor(loc, 1);
	glEnableVertexAttribArray(loc);

	loc = 3; /* a_rgba */
	glBindBuffer(GL_ARRAY_BUFFER, gui_vbo);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE,
			      sizeof(struct gui_quad),
			      (void *)offsetof(struct gui_quad, rgba));
	glVertexAttribDivisor(loc, 1);
	glEnableVertexAttribArray(loc);

	glBindVertexArray(0);
	shader_bind_quad(&(struct shader){.prog=gui_prg});
}

static void
gui_flush_draw_queue(void)
{
	glBufferData(GL_ARRAY_BUFFER, gui_count*sizeof(struct gui_quad), gui_data, GL_STREAM_DRAW);
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, gui_count);
	gui_total_count += gui_count;
	gui_draw_count++;
	gui_count = 0;
}

static int
gui_rect_overlap(struct gui_rect a, struct gui_rect b)
{
	return (a.x <= (b.x + b.w) && b.x <= (a.x + a.w))
		&& (a.y <= (b.y + b.h) && b.y <= (a.y + a.h));
}

static int
gui_mouse_in(struct gui_rect a)
{
	struct gui_rect m = { .x = xpos, .y = ypos, .w = 0, .h = 0 };
	return gui_rect_overlap(a, m);
}

static void
gui_push_quad(struct gui_quad q)
{
	gui_data[gui_count++] = q;
	if (gui_count == gui_nmax)
		gui_flush_draw_queue();
}

static float FW = 7.0;
static float FH = 9.0;

static void
gui_draw(void)
{
	int w, h;
	SDL_GL_GetDrawableSize(win_ctrl, &w, &h);
	struct gui_cmd *cmd;
	struct gui_rect r, clip = { 0, 0, w, h };
	struct gui_quad q = {
		.rgba = { 1.0, 1.0, 1.0, 1.0 },
	};
	GLint utex = glGetUniformLocation(gui_prg, "t_gui");
	glUseProgram(gui_prg);
	if (utex >= 0) {
		glActiveTexture(GL_TEXTURE0 + tex_gui.unit);
		glProgramUniform1i(gui_prg, utex, tex_gui.unit);
		glBindTexture(tex_gui.type, tex_gui.id);
	}
	glBindVertexArray(gui_vao);

	if (gui_cmd_queue_size == 0)
		return;

//	glScissor(clip.x, h - clip.y - clip.h, clip.w, clip.h);

	gui_for_each_cmd(cmd) {
		size_t i;
		float ox;
		char c;
		switch (cmd->type) {
		case GUI_CLIP:
			clip = cmd->clip;
			//glEnable(GL_SCISSOR_TEST);
			//glScissor(clip.x, h - clip.y - clip.h, clip.w, clip.h);
			break;
		case GUI_COLOR:
			q.rgba[0] = cmd->color.r / 255.0;
			q.rgba[1] = cmd->color.g / 255.0;
			q.rgba[2] = cmd->color.b / 255.0;
			q.rgba[3] = cmd->color.a / 255.0;
			break;
		case GUI_TEXT:
			ox = cmd->text.x;
			for (i = 0; i < cmd->text.len; i++, ox += FW) {
				c = cmd->text.str[i];
				q.xfrm[0] = +(FW+0.5)/(float)w;
				q.xfrm[1] = -(FH+0.5)/(float)h;
				q.xfrm[2] = -0.5 + ox/(float)w;
				q.xfrm[3] = +0.5 - cmd->text.y/(float)h;

				if (c <= ' ') continue;
				q.tfrm[0] = 1.0/16.0;
				q.tfrm[1] = 1.0/6.0;
				q.tfrm[2] = ((int)(c - ' ') % 16) / 16.0;
				q.tfrm[3] = ((int)(c - ' ') / 16) / 6.0;

				r.x = ox;
				r.y = cmd->text.y;
				r.w = FW;
				r.h = FH;
				if (gui_rect_overlap(r, clip))
					gui_push_quad(q);
			}
			break;
		case GUI_RECT:
			q.xfrm[0] = +((cmd->rect.w+0.5)/(float)w);
			q.xfrm[1] = -((cmd->rect.h+0.5)/(float)h);
			q.xfrm[2] = -0.5 + cmd->rect.x/(float)w;
			q.xfrm[3] = +0.5 - cmd->rect.y/(float)h;
			q.tfrm[0] = 0;
			q.tfrm[1] = 0;
			q.tfrm[2] = 0;
			q.tfrm[3] = 0;

			if (gui_rect_overlap(cmd->rect, clip))
				gui_push_quad(q);

			break;
		}
	}
	gui_flush_draw_queue();

//	glScissor(0, 0, w, h);
}

static void
gui_draw_grid_elem(int idx, int px, int py, size_t sz)
{
	if (&shaders[idx] == shader) {
		gui_color(255, 0, 0, 255);
		gui_rect(px-4, py-4, sz+8, sz+8);
	}
	if (idx < shader_count)
		gui_color(128,128,128,255);
	else
		gui_color(20,20,20,255);
	if (idx < shader_count && gui_mouse_in((struct gui_rect){px, py, sz, sz})) {
		gui_color(250,20,20,255);
		if (mouse_left_click())
			shader = &shaders[idx];
	}

	gui_rect(px, py, sz, sz);
	if (idx < shader_count) {
		char buf[] = "123467890";
		snprintf(buf, sizeof(buf), "%d", shaders[idx].prog);
		gui_rect(px, py, FW * strlen(buf), FH);

		gui_rect(px, py+sz, FW * strlen(shaders[idx].name), FH);
		gui_color(255, 255, 255,255);
		gui_text(px, py+sz, shaders[idx].name);
		gui_text(px, py, buf);
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
	char buf[64];
	size_t inst_nb = gui_total_count;
	size_t draw_nb = gui_draw_count;
#if 1 /* currently hacking on control window */
	render_window(win_live);
	SDL_GL_SwapWindow(win_live);
#endif
	render_window(win_ctrl);
	gui_begin();

//	gui_toggle("verbose", &verbose);
#if 0
//	gui_color(255,0,250,255);
//	gui_rect(xpos, ypos, 64, 64);
//	gui_clip(xpos, ypos, 200, 200);

	gui_color(210,10,10,255);
	gui_rect(0, 0, LEN(fps), 30);
	gui_color(255,255,255,255);

	float f, tf = 0;
	for (int i = 0; i < LEN(fps); i++) {
		tf += f = fps[(fps_count+i)%LEN(fps)];
		gui_rect(i, 0.25 * f, 1, 0.25 * MAX(120.0 - f, 0.0));
	}
	tf /= LEN(fps);
	snprintf(buf, sizeof(buf), "%2.0ffps", tf);
	gui_text(LEN(fps), 0, buf);
#endif
	gui_view_grid();
	gui_draw();
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
#if 0
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
	gui_init();
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
	float prev_time, curr_time;
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
		prev_time = curr_time;
		curr_time = get_time();

		input();
		for (i = 0; i < shader_count; i++)
			shader_poll(&shaders[i]);
		render();
		fps[fps_count++] = (1.0 / (curr_time - prev_time));
		fps_count %= LEN(fps);
	}
	fini();

	return 0;
}
