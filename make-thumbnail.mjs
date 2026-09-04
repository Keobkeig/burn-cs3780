/**
 * Render the project thumbnail from real model output.
 *
 * Four panels — a k-NN boundary, a trained MLP's decision surface, a trained
 * CNN's feature maps and a sinusoidal positional encoding — each computed by
 * the wasm bundle, then written as a PNG by hand so this needs no image
 * dependency.
 *
 *   bun make-thumbnail.mjs [out.png]
 */
import { deflateSync } from 'node:zlib';
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const WASM = '/Users/Programming/richie-portfolio/static/wasm/burn';
const OUT = process.argv[2] ?? '/Users/Programming/richie-portfolio/static/projects/burn-cs3780.png';

const burn = await import(pathToFileURL(`${WASM}/burn.js`).href);
await burn.default({ module_or_path: readFileSync(`${WASM}/burn_bg.wasm`) });

// Catppuccin Mocha, matching the site's default theme.
const C = {
	base: [30, 30, 46],
	crust: [17, 17, 27],
	surface1: [69, 71, 90],
	text: [205, 214, 244],
	blue: [137, 180, 250],
	red: [243, 139, 168],
	peach: [250, 179, 135],
	mauve: [203, 166, 247],
	green: [166, 227, 161],
	teal: [148, 226, 213]
};

const mix = (a, b, t) => {
	const k = Math.min(1, Math.max(0, t));
	return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
};

// ---------------------------------------------------------------------------
// Canvas
// ---------------------------------------------------------------------------

const GUTTER = 8;
const PANEL_W = 588;
const PANEL_H = 330;
const WIDTH = GUTTER * 3 + PANEL_W * 2;
const HEIGHT = GUTTER * 3 + PANEL_H * 2;

const pixels = new Uint8Array(WIDTH * HEIGHT * 3);
for (let i = 0; i < WIDTH * HEIGHT; i++) {
	pixels[i * 3] = C.crust[0];
	pixels[i * 3 + 1] = C.crust[1];
	pixels[i * 3 + 2] = C.crust[2];
}

function set(x, y, rgb) {
	if (x < 0 || y < 0 || x >= WIDTH || y >= HEIGHT) return;
	const o = (y * WIDTH + x) * 3;
	pixels[o] = rgb[0];
	pixels[o + 1] = rgb[1];
	pixels[o + 2] = rgb[2];
}

/** Bilinear-sample a res x res field and blit it into a panel. */
function blitField(panel, field, res, color) {
	const { x: ox, y: oy, w, h } = panel;
	for (let py = 0; py < h; py++) {
		for (let px = 0; px < w; px++) {
			// Field rows run bottom-up in data space.
			const fx = (px / (w - 1)) * (res - 1);
			const fy = (1 - py / (h - 1)) * (res - 1);
			const x0 = Math.floor(fx);
			const y0 = Math.floor(fy);
			const x1 = Math.min(res - 1, x0 + 1);
			const y1 = Math.min(res - 1, y0 + 1);
			const tx = fx - x0;
			const ty = fy - y0;
			const v =
				field[y0 * res + x0] * (1 - tx) * (1 - ty) +
				field[y0 * res + x1] * tx * (1 - ty) +
				field[y1 * res + x0] * (1 - tx) * ty +
				field[y1 * res + x1] * tx * ty;
			set(ox + px, oy + py, color(v));
		}
	}
}

function blitMatrix(panel, data, rows, cols, color) {
	const { x: ox, y: oy, w, h } = panel;
	for (let py = 0; py < h; py++) {
		for (let px = 0; px < w; px++) {
			const r = Math.min(rows - 1, Math.floor((py / h) * rows));
			const c = Math.min(cols - 1, Math.floor((px / w) * cols));
			set(ox + px, oy + py, color(data[r * cols + c]));
		}
	}
}

function dot(panel, bounds, px, py, rgb, radius = 4) {
	const { x: ox, y: oy, w, h } = panel;
	const cx = ox + ((px - bounds[0]) / (bounds[1] - bounds[0])) * w;
	const cy = oy + h - ((py - bounds[2]) / (bounds[3] - bounds[2])) * h;
	for (let dy = -radius - 1; dy <= radius + 1; dy++) {
		for (let dx = -radius - 1; dx <= radius + 1; dx++) {
			const d = Math.hypot(dx, dy);
			if (d > radius + 1) continue;
			// One pixel of feathering, and a dark rim so points read on any field.
			const rim = d > radius - 0.4 ? mix(rgb, C.crust, 0.55) : rgb;
			set(Math.round(cx + dx), Math.round(cy + dy), rim);
		}
	}
}

function boundsOf(points, pad = 0.06) {
	let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
	for (let i = 0; i < points.length / 2; i++) {
		x0 = Math.min(x0, points[i * 2]);
		x1 = Math.max(x1, points[i * 2]);
		y0 = Math.min(y0, points[i * 2 + 1]);
		y1 = Math.max(y1, points[i * 2 + 1]);
	}
	const px = (x1 - x0) * pad;
	const py = (y1 - y0) * pad;
	return [x0 - px, x1 + px, y0 - py, y1 + py];
}

const panels = [
	{ x: GUTTER, y: GUTTER, w: PANEL_W, h: PANEL_H },
	{ x: GUTTER * 2 + PANEL_W, y: GUTTER, w: PANEL_W, h: PANEL_H },
	{ x: GUTTER, y: GUTTER * 2 + PANEL_H, w: PANEL_W, h: PANEL_H },
	{ x: GUTTER * 2 + PANEL_W, y: GUTTER * 2 + PANEL_H, w: PANEL_W, h: PANEL_H }
];

// ---------------------------------------------------------------------------
// Panel 1 — k-NN on XOR
// ---------------------------------------------------------------------------

const RES = 160;
{
	// Low noise: at thumbnail size the four quadrants have to stay legible.
	const data = burn.sample_data(1, 80, 0.22, 11);
	const bounds = boundsOf(data.points);
	const r = burn.knn_boundary(data.points, data.labels, 5, 0, 0, RES, ...bounds);
	blitField(panels[0], r.grid, RES, (v) =>
		mix(C.base, Math.round(v) === 0 ? C.blue : C.red, 0.32)
	);
	for (let i = 0; i < data.labels.length; i++) {
		dot(panels[0], bounds, data.points[i * 2], data.points[i * 2 + 1],
			data.labels[i] === 0 ? C.blue : C.red);
	}
}

// ---------------------------------------------------------------------------
// Panel 2 — an MLP's decision surface on XOR
// ---------------------------------------------------------------------------

{
	const data = burn.sample_data(1, 120, 0.22, 13);
	const bounds = boundsOf(data.points);
	const r = burn.mlp_train(
		data.points, data.labels, new Uint32Array([16, 16]), 2, 500, 0.05, RES, ...bounds
	);
	blitField(panels[1], r.grid, RES, (v) =>
		mix(C.base, Math.round(v) === 0 ? C.mauve : C.peach, 0.34)
	);
	for (let i = 0; i < data.labels.length; i++) {
		dot(panels[1], bounds, data.points[i * 2], data.points[i * 2 + 1],
			data.labels[i] === 0 ? C.mauve : C.peach, 3);
	}
}

// ---------------------------------------------------------------------------
// Panel 3 — a trained CNN's feature maps
// ---------------------------------------------------------------------------

{
	const SIZE = 16;
	const PIXELS = SIZE * SIZE;
	const FILTERS = 12;
	const SHOWN = 8;
	const train = burn.shape_images(50, SIZE, 5);
	const model = new burn.CnnDemo(
		train.frames, SIZE, train.labels, 4,
		new Uint32Array([FILTERS]), 3, 40, 0.005, 16
	);

	// A ring: its feature maps show the most orientation structure.
	const index = [...train.labels].indexOf(2);
	const maps = model.feature_maps(train.frames.slice(index * PIXELS, (index + 1) * PIXELS));

	// Rank by activation energy and keep the liveliest.
	const ranked = Array.from({ length: FILTERS }, (_, f) => {
		const map = maps.slice(f * PIXELS, (f + 1) * PIXELS);
		return { f, energy: [...map].reduce((a, b) => a + b, 0) };
	})
		.sort((a, b) => b.energy - a.energy)
		.slice(0, SHOWN)
		.map((entry) => entry.f);

	// Four across, two down, with a small gap between tiles.
	const cols = 4;
	const rows = 2;
	const gap = 4;
	const tileW = Math.floor((panels[2].w - gap * (cols - 1)) / cols);
	const tileH = Math.floor((panels[2].h - gap * (rows - 1)) / rows);

	for (let py = 0; py < panels[2].h; py++) {
		for (let px = 0; px < panels[2].w; px++) set(panels[2].x + px, panels[2].y + py, C.crust);
	}

	for (const [slot, f] of ranked.entries()) {
		const map = maps.slice(f * PIXELS, (f + 1) * PIXELS);
		let max = 1e-6;
		for (const v of map) max = Math.max(max, v);
		const tile = {
			x: panels[2].x + (slot % cols) * (tileW + gap),
			y: panels[2].y + Math.floor(slot / cols) * (tileH + gap),
			w: tileW,
			h: tileH
		};
		blitMatrix(tile, map, SIZE, SIZE, (v) => mix(C.base, C.teal, v / max));
	}
	model.free();
}

// ---------------------------------------------------------------------------
// Panel 4 — sinusoidal positional encoding
// ---------------------------------------------------------------------------

{
	const seqLen = 40;
	const dModel = 72;
	const r = burn.positional_encoding(seqLen, dModel);
	blitMatrix(panels[3], r.grid, seqLen, dModel, (v) =>
		v < 0 ? mix(C.base, C.blue, -v * 0.85) : mix(C.base, C.peach, v * 0.85)
	);
}

// Hairline borders so the panels read as separate tiles.
for (const p of panels) {
	for (let x = -1; x <= p.w; x++) {
		set(p.x + x, p.y - 1, C.surface1);
		set(p.x + x, p.y + p.h, C.surface1);
	}
	for (let y = -1; y <= p.h; y++) {
		set(p.x - 1, p.y + y, C.surface1);
		set(p.x + p.w, p.y + y, C.surface1);
	}
}

// ---------------------------------------------------------------------------
// PNG encoding
// ---------------------------------------------------------------------------

const CRC_TABLE = (() => {
	const table = new Int32Array(256);
	for (let n = 0; n < 256; n++) {
		let c = n;
		for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
		table[n] = c;
	}
	return table;
})();

function crc32(buffer) {
	let c = 0xffffffff;
	for (const byte of buffer) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
	return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
	const length = Buffer.alloc(4);
	length.writeUInt32BE(data.length);
	const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
	const crc = Buffer.alloc(4);
	crc.writeUInt32BE(crc32(body));
	return Buffer.concat([length, body, crc]);
}

const ihdr = Buffer.alloc(13);
ihdr.writeUInt32BE(WIDTH, 0);
ihdr.writeUInt32BE(HEIGHT, 4);
ihdr[8] = 8; // bit depth
ihdr[9] = 2; // truecolor
// 10..12 stay zero: deflate, adaptive filtering, no interlace.

// One filter byte (0 = none) per scanline.
const raw = Buffer.alloc(HEIGHT * (1 + WIDTH * 3));
for (let y = 0; y < HEIGHT; y++) {
	const rowStart = y * (1 + WIDTH * 3);
	raw[rowStart] = 0;
	Buffer.from(pixels.buffer, y * WIDTH * 3, WIDTH * 3).copy(raw, rowStart + 1);
}

writeFileSync(
	OUT,
	Buffer.concat([
		Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
		chunk('IHDR', ihdr),
		chunk('IDAT', deflateSync(raw, { level: 9 })),
		chunk('IEND', Buffer.alloc(0))
	])
);

console.log(`wrote ${OUT} (${WIDTH}x${HEIGHT})`);
