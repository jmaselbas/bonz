#pragma once

void gui_begin(void *ctx);
size_t gui_size(void);
struct gui_state *gui_init(void *ctx, GLuint prog);
void gui_draw(int w, int h, GLuint prog, GLuint tex_s,  GLuint tex_c);
void gui_text(int x, int y, const char *s, uint8_t col);
uint8_t gui_color(uint8_t r, uint8_t g, uint8_t b);
void gui_fill(int x, int y, unsigned int w, unsigned int h, uint8_t c);

struct gui_rect {
	int16_t x, y;
	uint16_t w, h;
};

int gui_mouse_in(struct gui_rect a);

#include <stdarg.h>
#include <stdio.h>
static inline void gui_printf(int x, int y, char *fmt, ...)
{
	char buf[256];
	va_list ap;

	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);

	gui_text(x, y, buf, gui_color(255,255,255));
}

#ifdef GUI_IMPLEMENTATION
#undef offsetof
#define offsetof(type, memb) __builtin_offsetof(type, memb)

struct gui_quad {
	float pos_xfrm[4];
	float shp_xfrm[4];
	float img_xfrm[4];
};

struct color {
	union {
		uint8_t rgb[3];
		struct {
			uint8_t r, g, b;
		};
	};
};

struct gui_cmd {
	enum gui_cmd_type {
		GUI_SHAPE,
		GUI_TEXT,
	} type;
	union {
		struct {
			struct gui_rect rect;
			struct gui_rect shape;
			struct gui_rect image;
		} shape;
		struct {
			int16_t x, y;
			uint8_t col;
			uint8_t len;
			char str[];
		} text;
	};
};

struct gui_state {
	uint8_t last_color;
	size_t color_count;
	struct color colors[128];

	GLuint cmd_count;
	size_t cmd_queue_size;
	struct gui_cmd cmd_queue[4096];

	GLuint vao;
	union {
		GLuint vbo[2];
		struct {
			GLuint quad_vbo;
			GLuint inst_vbo;
		};
	};

	GLuint total_count;
	GLuint draw_count;
	GLuint quad_count;

	struct gui_quad quad[1024];
};
struct gui_state *gui;

size_t
gui_size(void)
{
	return sizeof(*gui);
}

void
gui_begin(void *ctx)
{
	gui = ctx;
	gui->cmd_count = 0;
	gui->cmd_queue_size = 0;
	gui->quad_count = 0;
	gui->total_count = 0;
	gui->draw_count = 0;
	gui->color_count = 0;
}

static size_t
gui_cmd_size(struct gui_cmd *cmd)
{
	size_t s = sizeof(*cmd);

	if (cmd->type == GUI_TEXT)
		s += cmd->text.len;

	return s;
}

#define gui_cmd_queue_end() (((void *)gui->cmd_queue) + gui->cmd_queue_size)

static struct gui_cmd *
gui_cmd_next(struct gui_cmd *cmd)
{
	if (cmd)
		cmd = ((void *)cmd) + gui_cmd_size(cmd);
	if (((void *)cmd) >= gui_cmd_queue_end())
		return NULL;
	return cmd;
}

#define gui_for_each_cmd(c) for ((c) = gui->cmd_queue; (c); (c) = gui_cmd_next(c))

void
gui_text(int x, int y, const char *s, uint8_t col)
{
	struct gui_cmd *cmd = gui_cmd_queue_end();
	size_t tlen = strlen(s);
	while (tlen > 0) {
		size_t len = tlen > UINT8_MAX ? UINT8_MAX : tlen;
		size_t size = len + sizeof(*cmd);
		if ((gui->cmd_queue_size + size) > sizeof(gui->cmd_queue)) {
			printf("!!\n");
			return;
		}
		cmd->type = GUI_TEXT;
		cmd->text.x = x;
		cmd->text.y = y;
		cmd->text.col = col;
		cmd->text.len = len;
		memcpy(cmd->text.str, s, len);

		tlen -= len;
		s += len;

		gui->cmd_queue_size += size;
		gui->cmd_count++;
	}
}

static void
gui_shape(struct gui_rect rect, struct gui_rect shape, struct gui_rect image)
{
	struct gui_cmd *cmd = gui_cmd_queue_end();
	size_t size = sizeof(*cmd);
	if ((gui->cmd_queue_size + size) > sizeof(gui->cmd_queue))
		return;
	cmd->type = GUI_SHAPE;
	cmd->shape.rect = rect;
	cmd->shape.shape = shape;
	cmd->shape.image = image;

	gui->cmd_queue_size += size;
	gui->cmd_count++;
}

static struct gui_rect
gui_rect(int x, int y, unsigned int w, unsigned int h)
{
	struct gui_rect rect = { .x = x, .y = y, .w = w, .h = h, };
	return rect;
}

static void
gui_image(struct gui_rect rect, struct gui_rect image)
{
	gui_shape(rect,
		  gui_rect(0, 0, 0, 0),
		  image);
}

void
gui_fill(int x, int y, unsigned int w, unsigned int h, uint8_t c)
{
	gui_shape(gui_rect(x, y, w, h),
		  gui_rect(0, 0, 0, 0),
		  gui_rect(c, 0, 0, 0));
}

uint8_t
gui_color(uint8_t r, uint8_t g, uint8_t b)
{
	size_t i;
	for (i = 0; i < gui->color_count && i < LEN(gui->colors); i++) {
		struct color c = gui->colors[i];
		if (c.r == r && c.g == g && c.b == b)
			return gui->last_color = i;
	}
	if (i < LEN(gui->colors)) {
		i = gui->color_count++;
		gui->colors[i].r = r;
		gui->colors[i].g = g;
		gui->colors[i].b = b;
		return gui->last_color = i;
	}
	return 0;
}

struct gui_state *
gui_init(void *ctx, GLuint prog)
{
	static float quad[] = {
		0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0,
	};
	GLuint loc;

	gui_begin(ctx);
	glUseProgram(prog);

	glGenVertexArrays(1, &gui->vao);
	glBindVertexArray(gui->vao);

	glGenBuffers(2, &gui->vbo[0]);

	glBindBuffer(GL_ARRAY_BUFFER, gui->quad_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, gui->inst_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(gui->quad), gui->quad, GL_STREAM_DRAW);


	loc = 0; /* a_pos */
	glBindBuffer(GL_ARRAY_BUFFER, gui->quad_vbo);
	glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glVertexAttribDivisor(loc, 0);
	glEnableVertexAttribArray(loc);

	loc = 1; /* a_pos_xfrm */
	glBindBuffer(GL_ARRAY_BUFFER, gui->inst_vbo);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE,
			      sizeof(struct gui_quad),
			      (void *)offsetof(struct gui_quad, pos_xfrm));
	glVertexAttribDivisor(loc, 1);
	glEnableVertexAttribArray(loc);

	loc = 2; /* a_shp_xfrm */
	glBindBuffer(GL_ARRAY_BUFFER, gui->inst_vbo);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE,
			      sizeof(struct gui_quad),
			      (void *)offsetof(struct gui_quad, shp_xfrm));
	glVertexAttribDivisor(loc, 1);
	glEnableVertexAttribArray(loc);

	loc = 3; /* a_img_xfrm */
	glBindBuffer(GL_ARRAY_BUFFER, gui->inst_vbo);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE,
			      sizeof(struct gui_quad),
			      (void *)offsetof(struct gui_quad, img_xfrm));
	glVertexAttribDivisor(loc, 1);
	glEnableVertexAttribArray(loc);

	glBindVertexArray(0);


//	glBindTexture(gui->tex_color.type, gui->tex_color.id);
//	glTexParameteri(gui->tex_color.type, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	glTexParameteri(gui->tex_color.type, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	return gui;
}

static void
gui_flush_draw_queue(void)
{
	size_t s = gui->quad_count * sizeof(*gui->quad);

	glBindBuffer(GL_ARRAY_BUFFER, gui->inst_vbo);
	glBufferData(GL_ARRAY_BUFFER, s, gui->quad, GL_STREAM_DRAW);
	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, gui->quad_count);
	gui->total_count += gui->quad_count;
	gui->draw_count++;
	gui->quad_count = 0;
}

static int
gui_rect_overlap(struct gui_rect a, struct gui_rect b)
{
	return (a.x <= (b.x + b.w) && b.x <= (a.x + a.w))
		&& (a.y <= (b.y + b.h) && b.y <= (a.y + a.h));
}

int
gui_mouse_in(struct gui_rect a)
{
	struct gui_rect m = { .x = xpos, .y = ypos, .w = 0, .h = 0 };
	return gui_rect_overlap(a, m);
}

static void
gui_push_quad(struct gui_quad q)
{
	gui->quad[gui->quad_count++] = q;
	if (gui->quad_count == LEN(gui->quad))
		gui_flush_draw_queue();
}

#if 0
static float FW = 7.0 * 2.0;
static float FH = 9.0 * 2.0;

void
gui_draw(void)
{
	int w = g_state->width, h = g_state->height;
	struct gui_cmd *cmd;
	struct gui_rect r, clip = { 0, 0, w, h };
	struct gui_quad q;
	struct shader *s = game_get_shader(g_asset, SHADER_GUI);
	GLuint prog = s->prog;
	GLint utex;
	struct texture *tex_color = &gui->tex_color;
	struct texture *tex_shape = game_get_texture(g_asset, TEXTURE_GUI_SHAPE);

	if (gui->cmd_queue_size == 0)
		return;

	glUseProgram(prog);

	utex = glGetUniformLocation(prog, "t_shape");
	if (utex >= 0 && tex_shape != NULL) {
		uint8_t b[3] = { 0xff, 0xff, 0xff };
		glActiveTexture(GL_TEXTURE0 + 1);
		glUniform1i(utex, 1);
		glBindTexture(tex_shape->type, tex_shape->id);
		glTexSubImage2D(tex_shape->type, 0, 0, 0, 1, 1, GL_RGB, GL_UNSIGNED_BYTE, &b);
	}

	utex = glGetUniformLocation(prog, "t_color");
	if (utex >= 0 && tex_color != NULL) {
		glActiveTexture(GL_TEXTURE0 + 2);
		glUniform1i(utex, 2);
		glBindTexture(tex_color->type, tex_color->id);
		glTexSubImage2D(tex_color->type, 0, 0, 0, 128, 1, GL_RGB, GL_UNSIGNED_BYTE, gui->colors);
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
#endif
#endif
