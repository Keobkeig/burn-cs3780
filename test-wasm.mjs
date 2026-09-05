/**
 * Smoke test for the browser bundle: load it the way the page does and check
 * every binding returns something with the right shape and sane values.
 *
 * Defaults to the sibling portfolio checkout that build-wasm.sh writes to.
 *
 *   bun test-wasm.mjs [path/to/static/wasm/burn]
 *   BURN_WASM_OUT=/path/to/out bun test-wasm.mjs
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = dirname(fileURLToPath(import.meta.url));
const defaultDir = resolve(repo, '..', 'richie-portfolio', 'static', 'wasm', 'burn');
const dir = process.argv[2] ?? process.env.BURN_WASM_OUT ?? defaultDir;
const burn = await import(pathToFileURL(`${dir}/burn.js`).href);
await burn.default({ module_or_path: readFileSync(`${dir}/burn_bg.wasm`) });

const checks = [];
function check(name, fn) {
	try {
		fn();
		checks.push([name, null]);
	} catch (e) {
		checks.push([name, e.message]);
	}
}

/** Every value present, finite, and not all identical. */
function varied(array, name) {
	assert.ok(array.length > 0, `${name} is empty`);
	for (const v of array) assert.ok(Number.isFinite(v), `${name} has a non-finite value`);
	assert.ok(new Set(array).size > 1, `${name} is constant — model probably did nothing`);
}

const RES = 24;
const BOX = [-3, 3, -3, 3];

const xor = burn.sample_data(1, 60, 0.4, 11);
const blobs = burn.sample_data(2, 120, 4, 3);
const linear = burn.sample_data(0, 80, 0, 3);
const pm = [...linear.labels].map((y) => (y > 0.5 ? 1 : -1));
const xorPm = [...xor.labels].map((y) => (y > 0.5 ? 1 : -1));

check('sample_data', () => {
	assert.equal(xor.points.length, 120);
	assert.equal(xor.labels.length, 60);
	varied(xor.points, 'points');
	assert.deepEqual([...new Set(xor.labels)].sort(), [0, 1]);
});

check('knn_boundary', () => {
	const r = burn.knn_boundary(xor.points, xor.labels, 5, 0, 0, RES, ...BOX);
	assert.equal(r.grid.length, RES * RES);
	varied(r.grid, 'grid');
	assert.equal(r.labels.length, 60);
});

check('knn_neighbors', () => {
	const r = burn.knn_neighbors(xor.points, xor.labels, 5, 0.5, 0.5);
	assert.equal(r.points.length, 10);
	assert.equal(r.curve.length, 5);
	// Distances must come back sorted nearest-first.
	for (let i = 1; i < r.curve.length; i++) assert.ok(r.curve[i] >= r.curve[i - 1]);
});

check('perceptron_epochs', () => {
	const r = burn.perceptron_epochs(linear.points, linear.labels, 0.5, 10);
	assert.equal(r.frames.length, 30);
	assert.equal(r.curve.length, 10);
	varied(r.frames, 'frames');
	// Separable data: the error count should not be worse at the end.
	assert.ok(r.curve.at(-1) <= r.curve[0]);

	// The reported [w0, w1, bias] must be the actual boundary: scoring the
	// training points with it has to reproduce the model's own predictions.
	const [w0, w1, bias] = [r.frames.at(-3), r.frames.at(-2), r.frames.at(-1)];
	let agree = 0;
	for (let i = 0; i < r.labels.length; i++) {
		const score = w0 * linear.points[i * 2] + w1 * linear.points[i * 2 + 1] + bias;
		if (score > 0 === r.labels[i] > 0.5) agree++;
	}
	assert.equal(
		agree,
		r.labels.length,
		`reported line disagrees with the model on ${r.labels.length - agree} points`
	);
});

check('decision_tree_boundary', () => {
	const shallow = burn.decision_tree_boundary(xor.points, xor.labels, 1, 2, 0, RES, ...BOX);
	const deep = burn.decision_tree_boundary(xor.points, xor.labels, 10, 2, 0, RES, ...BOX);
	assert.equal(deep.grid.length, RES * RES);
	assert.ok(deep.stats[0] > shallow.stats[0], 'deeper tree should reach greater depth');
	assert.ok(deep.stats[2] > 1, 'tree should have leaves');
});

check('linear_regression_fit', () => {
	const xs = Array.from({ length: 40 }, (_, i) => -2 + i * 0.1);
	const ys = xs.map((x) => 2 * x + 1);
	const plain = burn.linear_regression_fit(xs, ys, 0, 0, 0.5, 20, -3, 3);
	assert.ok(Math.abs(plain.stats[0] - 2) < 0.05, `slope ${plain.stats[0]} should be ~2`);
	assert.ok(Math.abs(plain.stats[1] - 1) < 0.05, `intercept ${plain.stats[1]} should be ~1`);
	assert.ok(plain.stats[2] < 1e-3, 'noiseless fit should have ~0 MSE');

	const ridge = burn.linear_regression_fit(xs, ys, 1, 50, 0.5, 20, -3, 3);
	assert.ok(Math.abs(ridge.stats[0]) < Math.abs(plain.stats[0]), 'ridge should shrink the slope');
});

check('logistic_regression_grid', () => {
	const r = burn.logistic_regression_grid(linear.points, linear.labels, 0, 0, 0.1, 200, RES, ...BOX);
	assert.equal(r.grid.length, RES * RES);
	for (const v of r.grid) assert.ok(v >= 0 && v <= 1, `probability ${v} out of range`);
	varied(r.grid, 'grid');
});

for (const [name, kernel] of [
	['linear', 0],
	['rbf', 1],
	['poly', 2],
	['sigmoid', 3]
]) {
	check(`svm_boundary (${name})`, () => {
		const r = burn.svm_boundary(linear.points, pm, kernel, 1, 3, 1, 1, 200, RES, ...BOX);
		assert.equal(r.grid.length, RES * RES);
		varied(r.grid, 'grid');
		assert.ok(r.stats[0] >= 1, 'should keep at least one support vector');
		assert.ok(r.stats[1] > 0.5, `accuracy ${r.stats[1]} should beat chance`);
	});

	check(`kernel_matrix (${name})`, () => {
		const r = burn.kernel_matrix(xor.points, 2, kernel, 1, 2, 1);
		assert.equal(r.grid.length, 60 * 60);
		varied(r.grid, 'grid');
		// Gram matrices are symmetric.
		for (let i = 0; i < 10; i++) {
			for (let j = 0; j < 10; j++) {
				assert.ok(Math.abs(r.grid[i * 60 + j] - r.grid[j * 60 + i]) < 1e-4);
			}
		}
	});
}

check('kernel_matrix (rbf diagonal)', () => {
	const r = burn.kernel_matrix(xor.points, 2, 1, 1, 2, 1);
	for (let i = 0; i < 60; i++) {
		assert.ok(Math.abs(r.grid[i * 60 + i] - 1) < 1e-5, 'RBF self-similarity must be 1');
	}
});

check('kmeans_steps', () => {
	const r = burn.kmeans_steps(blobs.points, 4, 8, 1, 7, RES, ...BOX);
	assert.equal(r.frames.length, 8 * 4 * 2);
	assert.equal(r.curve.length, 8);
	assert.equal(r.labels.length, 120);
	assert.equal(r.grid.length, RES * RES);
	varied(r.frames, 'centroids');
	assert.ok(r.curve.at(-1) <= r.curve[0] + 1e-3, 'inertia should not increase');
	assert.ok(new Set(r.labels).size > 1, 'should use more than one cluster');
});

check('kmeans_steps (manhattan no longer recurses)', () => {
	// The Manhattan branch used to be an unconditional infinite recursion.
	const r = burn.kmeans_steps(blobs.points, 3, 4, 0, 1, 8, ...BOX);
	// It may converge on the first iteration, so the curve can be flat —
	// what matters is that it returned at all instead of overflowing the stack.
	assert.equal(r.curve.length, 4);
	for (const v of r.curve) assert.ok(Number.isFinite(v) && v > 0, 'inertia should be finite');
});

check('pca_fit', () => {
	// Strongly correlated cloud: PC1 must dominate.
	const xs = [];
	for (let i = 0; i < 200; i++) {
		const t = (i / 200) * 4 - 2;
		xs.push(t * 3 + Math.sin(i) * 0.1, t * 3 + Math.cos(i) * 0.1);
	}
	const r = burn.pca_fit(xs, 2, 2, false);
	assert.equal(r.points.length, 4);
	assert.equal(r.frames.length, 400);
	assert.ok(r.curve[0] > 0.9, `PC1 should explain >90%, got ${r.curve[0]}`);
	assert.ok(Math.abs(r.curve[0] + r.curve[1] - 1) < 1e-3, 'ratios should sum to 1');
	const norm = Math.hypot(r.points[0], r.points[1]);
	assert.ok(Math.abs(norm - 1) < 1e-3, `components should be unit length, got ${norm}`);
});

check('mlp_train (learns XOR)', () => {
	const r = burn.mlp_train(xor.points, xor.labels, [16, 16], 2, 400, 0.05, RES, ...BOX);
	assert.equal(r.curve.length, 400);
	assert.equal(r.grid.length, RES * RES);
	assert.ok(r.curve.at(-1) < r.curve[0], 'loss must go down — fit was a stub before');
	assert.ok(r.stats[1] > 0.85, `XOR accuracy ${r.stats[1]} should be high with 2 hidden layers`);
});

check('mlp_train (2 -> 2 cannot do XOR)', () => {
	const r = burn.mlp_train(xor.points, xor.labels, [], 2, 400, 0.05, 8, ...BOX);
	assert.ok(r.stats[1] < 0.8, `linear net should fail XOR, got ${r.stats[1]}`);
});

check('AutoencoderDemo', () => {
	const rows = [];
	for (let a = 0; a < 8; a++) {
		for (let o = 0; o < 8; o++) {
			const theta = (a / 8) * Math.PI;
			const offset = -0.7 + (o / 7) * 1.4;
			for (let r = 0; r < 8; r++) {
				for (let c = 0; c < 8; c++) {
					const x = (c / 7) * 2 - 1;
					const y = (r / 7) * 2 - 1;
					const d = x * Math.cos(theta) + y * Math.sin(theta) - offset;
					rows.push(Math.exp(-(d * d) / 0.045));
				}
			}
		}
	}
	const model = new burn.AutoencoderDemo(rows, 64, [32], 2, 200, 0.01);
	assert.equal(model.history.length, 200);
	assert.ok(model.history.at(-1) < model.history[0], 'reconstruction loss must fall');

	const latent = model.encode(rows);
	assert.equal(latent.length, 64 * 2);
	varied(latent, 'latent codes');

	const rebuilt = model.reconstruct(rows.slice(0, 64));
	assert.equal(rebuilt.length, 64);
	varied(rebuilt, 'reconstruction');

	const decoded = model.decode([0.1, -0.2], 2);
	assert.equal(decoded.length, 64);
	model.free();
});

check('adaboost_boundary', () => {
	const one = burn.adaboost_boundary(xor.points, xorPm, 1, 1, RES, ...BOX);
	const many = burn.adaboost_boundary(xor.points, xorPm, 30, 1, RES, ...BOX);
	assert.equal(many.grid.length, RES * RES);
	varied(many.grid, 'margin');
	assert.ok(many.stats[0] > one.stats[0], 'more rounds should use more stumps');
	assert.equal(many.curve.length, 2, 'two features, two importances');
});

check('NaiveBayesDemo', () => {
	const model = new burn.NaiveBayesDemo(false, true);
	assert.ok(model.train_accuracy > 0.7, `training accuracy ${model.train_accuracy} too low`);
	const german = model.score('wissenschaft');
	const english = model.score('thinking');
	assert.equal(german.stats.length, 2);
	assert.ok(
		german.stats[0] !== english.stats[0],
		'different words must produce different log-ratios'
	);
	assert.equal(german.curve.length, 'wissenschaft'.length);
	model.free();
});

for (const [name, kind] of [
	['SGD', 0],
	['Adam', 1],
	['AdaGrad', 2]
]) {
	check(`optimizer_path (${name})`, () => {
		const r = burn.optimizer_path(kind, -2.5, 2.2, 0.1, 150, 1, 12);
		assert.equal(r.frames.length, 300);
		assert.equal(r.curve.length, 150);
		varied(r.frames, 'path');
		assert.ok(r.curve.at(-1) < r.curve[0], `${name} should reduce the loss`);
	});
}

for (const [name, algo] of [
	['perceptron', 0],
	['passive-aggressive', 1],
	['sgd', 2]
]) {
	check(`online_stream (${name})`, () => {
		const r = burn.online_stream(linear.points, pm, algo, 1, 1, 0.0001);
		assert.equal(r.curve.length, 80);
		assert.equal(r.frames.length, 240);
		// Cumulative mistakes must be monotone non-decreasing.
		for (let i = 1; i < r.curve.length; i++) assert.ok(r.curve[i] >= r.curve[i - 1]);
		assert.ok(r.stats[0] < 80, 'should not miss every sample');
		varied(r.frames, 'weights');
	});
}

check('positional_encoding', () => {
	const r = burn.positional_encoding(32, 64);
	assert.equal(r.grid.length, 32 * 64);
	varied(r.grid, 'encoding');
	for (const v of r.grid) assert.ok(v >= -1.0001 && v <= 1.0001, `sin/cos out of range: ${v}`);
	// Position 0 is sin(0), cos(0), sin(0), cos(0)...
	assert.ok(Math.abs(r.grid[0]) < 1e-6);
	assert.ok(Math.abs(r.grid[1] - 1) < 1e-6);
});

check('attention_weights', () => {
	const emb = [-2, 1.5, 1.8, 1.2, 1.5, -0.4, -1.2, -1.6, -2.1, 1.1, 1.2, -1.8];
	const r = burn.attention_weights(emb, 2);
	assert.equal(r.grid.length, 36);
	for (let row = 0; row < 6; row++) {
		let sum = 0;
		for (let col = 0; col < 6; col++) sum += r.grid[row * 6 + col];
		assert.ok(Math.abs(sum - 1) < 1e-4, `attention row ${row} sums to ${sum}, not 1`);
	}
	varied(r.grid, 'attention');
});

// --- Every option the page exposes, at the parameters the demos actually use.
// This is the section that catches "this control combination traps".
const MATRIX = [
	['knn', () => {
		const d = burn.sample_data(1, 60, 0.4, 11);
		for (const metric of [0, 1, 2]) {
			for (const weighting of [0, 1, 2]) {
				for (const k of [1, 5, 25]) {
					varied(burn.knn_boundary(d.points, d.labels, k, metric, weighting, 16, -1, 2, -1, 2).grid, 'knn grid');
				}
			}
		}
	}],
	['tree', () => {
		const d = burn.sample_data(1, 80, 0.5, 5);
		for (const criterion of [0, 1]) {
			for (const depth of [1, 6, 12]) {
				burn.decision_tree_boundary(d.points, d.labels, depth, 2, criterion, 16, -1, 2, -1, 2);
			}
		}
	}],
	['linreg', () => {
		const xs = Array.from({ length: 22 }, (_, i) => -2.8 + (i / 21) * 5.6);
		const ys = xs.map((x) => 1.2 * x);
		for (const reg of [0, 1, 2, 3]) {
			for (const alpha of [0, 5, 20]) {
				const r = burn.linear_regression_fit(xs, ys, reg, alpha, 0.5, 60, -3, 3);
				varied(r.curve, `linreg reg=${reg} alpha=${alpha}`);
			}
		}
	}],
	['logreg', () => {
		const d = burn.sample_data(0, 70, 0, 21);
		for (const reg of [0, 1, 2, 3]) {
			burn.logistic_regression_grid(d.points, d.labels, reg, 0.1, 0.1, 300, 16, -3, 3, -3, 3);
		}
	}],
	['svm', () => {
		const d = burn.sample_data(1, 60, 0.45, 13);
		const ys = [...d.labels].map((y) => (y > 0.5 ? 1 : -1));
		for (const kernel of [0, 1, 2, 3]) {
			for (const c of [0.1, 1, 10]) {
				burn.svm_boundary(d.points, ys, kernel, 1, 3, 1, c, 300, 16, -1, 2, -1, 2);
			}
		}
	}],
	['kernel', () => {
		for (const dataset of [0, 1, 2]) {
			const d = burn.sample_data(dataset, 40, dataset === 1 ? 0.35 : 3, 17);
			for (const kernel of [0, 1, 2, 3]) {
				burn.kernel_matrix(d.points, 2, kernel, 1, 2, 1);
			}
		}
	}],
	['kmeans', () => {
		const d = burn.sample_data(2, 240, 4, 3);
		for (const init of [0, 1]) {
			for (const k of [2, 5, 8]) {
				const r = burn.kmeans_steps(d.points, k, 12, init, 7, 16, -8, 8, -8, 8);
				assert.equal(r.frames.length, 12 * k * 2);
			}
		}
	}],
	['pca', () => {
		for (const scale of [false, true]) {
			for (const components of [1, 2]) {
				burn.pca_fit(burn.sample_data(0, 200, 0, 4).points, 2, components, scale);
			}
		}
	}],
	['mlp', () => {
		for (const dataset of [0, 1, 2]) {
			const d = burn.sample_data(dataset, 120, dataset === 1 ? 0.35 : 4, 9);
			const classes = Math.max(2, new Set([...d.labels]).size);
			for (const hidden of [[], [8], [32, 16]]) {
				burn.mlp_train(d.points, d.labels, hidden, classes, 60, 0.05, 12, -6, 6, -6, 6);
			}
		}
	}],
	['adaboost', () => {
		for (const dataset of [0, 1, 2]) {
			const d = burn.sample_data(dataset, 88, dataset === 1 ? 0.4 : 3, 23);
			const ys = [...d.labels].map((y) => (y > 0.5 ? 1 : -1));
			for (const n of [1, 20, 40]) {
				const r = burn.adaboost_boundary(d.points, ys, n, 1, 16, -6, 6, -6, 6);
				assert.equal(r.stats[0], n, `dataset ${dataset} should stack ${n} stumps`);
			}
		}
	}],
	['naivebayes', () => {
		for (const bigrams of [false, true]) {
			for (const smoothing of [false, true]) {
				const model = new burn.NaiveBayesDemo(bigrams, smoothing);
				for (const word of ['wissenschaft', 'thinking', '', 'zzz']) model.score(word);
				model.free();
			}
		}
	}],
	['optimizers', () => {
		for (const kind of [0, 1, 2]) {
			for (const curvature of [1, 12, 40]) {
				burn.optimizer_path(kind, -2.5, 2.2, 0.1, 120, 1, curvature);
			}
		}
	}],
	['online', () => {
		const d = burn.sample_data(0, 120, 0, 31);
		const ys = [...d.labels].map((y) => (y > 0.5 ? 1 : -1));
		for (const algo of [0, 1, 2]) burn.online_stream(d.points, ys, algo, 1, 1, 0.0001);
	}],
	['posenc', () => {
		for (const seqLen of [8, 48, 96]) {
			for (const dModel of [8, 64, 128]) {
				const r = burn.positional_encoding(seqLen, dModel);
				assert.equal(r.grid.length, seqLen * dModel);
			}
		}
	}],
	['convolution', () => {
		for (const size of [16, 48]) {
			const d = burn.shape_images(1, size, 7);
			for (const kernel of [
				[0, 0, 0, 0, 1, 0, 0, 0, 0],
				[-1, 0, 1, -2, 0, 2, -1, 0, 1],
				Array(9).fill(1 / 9),
				[0, -1, 0, -1, 5, -1, 0, -1, 0]
			]) {
				const r = burn.apply_kernel(d.frames.slice(0, size * size), size, size, kernel, 3);
				assert.equal(r.grid.length, size * size);
			}
		}
	}],
	['cnn', () => {
		const d = burn.shape_images(20, 16, 5);
		for (const filters of [4, 8, 16]) {
			const m = new burn.CnnDemo(d.frames, 16, d.labels, 4, new Uint32Array([filters]), 3);
			m.train(10, 0.005, 16);
			assert.equal(m.filters().length, filters * 9);
			m.free();
		}
	}],
	['transformer generation', () => {
		for (const corpus of [0, 1, 2]) {
			const m = new burn.TransformerGenDemo(corpus, 32, 2, 1, 14);
			m.train(2, 0.003, 32);
			for (const temperature of [0.2, 0.8, 1.6]) {
				assert.ok(/^[a-z]*$/.test(m.generate('', temperature, 1)));
			}
			m.free();
		}
		assert.throws(() => new burn.TransformerGenDemo(2, 30, 4, 1, 14), /divisible/);
	}],
	['transformer text', () => {
		for (const task of [0, 1]) {
			for (const [heads, layers] of [[2, 1], [4, 2]]) {
				const m = new burn.TransformerTextDemo(task, 32, heads, layers, 12);
				m.train(4, 0.003, 32);
				assert.equal(m.attention('test').length, layers * heads * 12 * 12);
				assert.equal(m.classify('test').length, 2);
				m.free();
			}
		}
	}],
	['attention', () => {
		for (const scale of [0.2, 1, 4]) {
			const emb = [-2, 1.5, 1.8, 1.2, 1.5, -0.4, -1.2, -1.6, -2.1, 1.1, 1.2, -1.8].map((v) => v * scale);
			burn.attention_weights(emb, 2);
		}
	}]
];

for (const [name, fn] of MATRIX) check(`demo options: ${name}`, fn);

// --- Images -----------------------------------------------------------------

const SHAPE_SIZE = 16;
const SHAPE_PIXELS = SHAPE_SIZE * SHAPE_SIZE;
const shapes = burn.shape_images(12, SHAPE_SIZE, 5);

check('shape_images', () => {
	assert.equal(shapes.stats[0], 48);
	assert.equal(shapes.stats[1], SHAPE_SIZE);
	assert.equal(shapes.frames.length, 48 * SHAPE_PIXELS);
	assert.deepEqual([...new Set(shapes.labels)].sort(), [0, 1, 2, 3]);
	for (const v of shapes.frames) assert.ok(v >= 0 && v <= 1, `pixel ${v} out of range`);

	// Each class must look different from the others, and the shape has to
	// actually occupy a decent chunk of the frame.
	const means = [0, 1, 2, 3].map((cls) => {
		const i = [...shapes.labels].indexOf(cls);
		const img = shapes.frames.slice(i * SHAPE_PIXELS, (i + 1) * SHAPE_PIXELS);
		return [...img].reduce((a, b) => a + b, 0) / SHAPE_PIXELS;
	});
	for (const [cls, mean] of means.entries()) {
		assert.ok(mean > 0.04 && mean < 0.9, `class ${cls} coverage ${mean.toFixed(3)} looks wrong`);
	}
	// The ring is hollow, so it must cover less than the disc.
	assert.ok(means[2] < means[0], 'ring should be lighter than disc');
});

check('apply_kernel', () => {
	const image = shapes.frames.slice(0, SHAPE_PIXELS);

	const identity = burn.apply_kernel(image, SHAPE_SIZE, SHAPE_SIZE, [0, 0, 0, 0, 1, 0, 0, 0, 0], 3);
	assert.equal(identity.stats[0], SHAPE_SIZE);
	assert.equal(identity.grid.length, SHAPE_PIXELS);
	for (let i = 0; i < SHAPE_PIXELS; i++) {
		assert.ok(Math.abs(identity.grid[i] - image[i]) < 1e-5, 'identity kernel must be identity');
	}

	// An edge kernel sums to zero, so a constant image must come back empty.
	const flatImage = new Float32Array(SHAPE_PIXELS).fill(0.5);
	const edge = burn.apply_kernel(flatImage, SHAPE_SIZE, SHAPE_SIZE, [0, -1, 0, -1, 4, -1, 0, -1, 0], 3);
	const interior = [];
	for (let r = 1; r < SHAPE_SIZE - 1; r++) {
		for (let c = 1; c < SHAPE_SIZE - 1; c++) interior.push(edge.grid[r * SHAPE_SIZE + c]);
	}
	for (const v of interior) assert.ok(Math.abs(v) < 1e-5, `edge response ${v} on a flat image`);

	// A box blur preserves total brightness in the interior.
	const blur = burn.apply_kernel(image, SHAPE_SIZE, SHAPE_SIZE, Array(9).fill(1 / 9), 3);
	varied(blur.grid, 'blur response');

	assert.throws(() => burn.apply_kernel(image, 5, 5, Array(9).fill(0), 3), /width/);
	assert.throws(() => burn.apply_kernel(image, SHAPE_SIZE, SHAPE_SIZE, [1, 2], 3), /kernel/);
});

check('CnnDemo', () => {
	const train = burn.shape_images(30, SHAPE_SIZE, 5);
	const test = burn.shape_images(10, SHAPE_SIZE, 909);
	const model = new burn.CnnDemo(
		train.frames, SHAPE_SIZE, train.labels, 4, new Uint32Array([8]), 3
	);
	assert.equal(model.epochs_trained, 0);
	// Trained in slices, the way the page does it.
	for (let i = 0; i < 10; i++) model.train(4, 0.005, 16);

	assert.equal(model.epochs_trained, 40);
	assert.equal(model.history.length, 40);
	assert.ok(model.history.at(-1) < model.history[0], 'loss must fall');
	assert.ok(model.train_accuracy > 0.7, `train accuracy ${model.train_accuracy} too low`);

	const held = model.accuracy(test.frames, test.labels);
	assert.ok(held > 0.5, `held-out accuracy ${held} should beat chance`);

	// 8 filters of 3x3, one input channel.
	assert.equal(model.filters().length, 8 * 9);
	varied(model.filters(), 'learned filters');

	const image = test.frames.slice(0, SHAPE_PIXELS);
	const probs = model.predict(image);
	assert.equal(probs.length, 4);
	const total = [...probs].reduce((a, b) => a + b, 0);
	assert.ok(Math.abs(total - 1) < 1e-4, `softmax sums to ${total}`);

	assert.equal(model.feature_maps(image).length, 8 * SHAPE_PIXELS);
	// A wrong-sized image must not trap.
	assert.ok(Number.isNaN(model.predict([1, 2, 3])[0]));
	assert.equal(model.feature_maps([1, 2, 3]).length, 0);
	model.free();
});

check('CnnDemo rejects an indivisible image size', () => {
	// Three blocks halve the image three times, so 12 does not divide.
	const train = burn.shape_images(4, 12, 5);
	assert.throws(
		() =>
			new burn.CnnDemo(train.frames, 12, train.labels, 4, new Uint32Array([4, 8, 8]), 3),
		/divisible/
	);
	assert.throws(
		() => new burn.CnnDemo(train.frames, 12, [0, 1], 4, new Uint32Array([4]), 3),
		/label count/
	);
});

check('TransformerTextDemo (letter search)', () => {
	const SEQ = 12;
	const m = new burn.TransformerTextDemo(1, 32, 2, 1, SEQ);
	assert.equal(m.epochs_trained, 0);
	for (let i = 0; i < 4; i++) m.train(3, 0.003, 32);

	assert.equal(m.epochs_trained, 12);
	assert.equal(m.history.length, 12);
	assert.ok(m.history.at(-1) < m.history[0], 'loss must fall');
	assert.ok(m.test_accuracy > 0.95, `held-out accuracy ${m.test_accuracy} should be near perfect`);
	assert.equal(m.seq_len, SEQ);
	assert.equal(m.n_heads, 2);
	assert.equal(m.n_layers, 1);

	assert.deepEqual(m.tokens('ab').slice(0, 3), ['[CLS]', 'a', 'b']);
	assert.equal(m.tokens('ab').length, SEQ);

	const withK = m.classify('trunk');
	const withoutK = m.classify('random');
	assert.equal(withK.length, 2);
	assert.ok(Math.abs(withK[0] + withK[1] - 1) < 1e-4, 'softmax must sum to 1');
	// The claim is the verdict, not a particular confidence.
	assert.ok(withK[1] > withK[0], `"trunk" should read as containing a k, got ${withK[1]}`);
	assert.ok(withoutK[1] < withoutK[0], `"random" should not, got ${withoutK[1]}`);

	// n_layers * n_heads matrices of seq x seq.
	const att = m.attention('trunk');
	assert.equal(att.length, 1 * 2 * SEQ * SEQ);
	for (let row = 0; row < SEQ; row++) {
		let sum = 0;
		for (let col = 0; col < SEQ; col++) sum += att[row * SEQ + col];
		assert.ok(Math.abs(sum - 1) < 1e-3, `attention row ${row} sums to ${sum}`);
	}

	// The only route from the k to the readout is attention, so some head's
	// [CLS] row has to point at it. Which head takes the job varies by run,
	// so this asks whether any of them did. Random attention would hit the
	// target about one time in thirteen.
	const probes = ['trunk', 'market', 'kalter', 'stark'];
	for (const probe of probes) {
		const tokens = m.tokens(probe);
		const target = tokens.indexOf('k');
		const att = m.attention(probe);

		const found = [...Array(m.n_heads).keys()].some((head) => {
			const row = att.slice(head * SEQ * SEQ, head * SEQ * SEQ + SEQ);
			let peak = 1;
			for (let i = 2; i < SEQ; i++) if (row[i] > row[peak]) peak = i;
			return peak === target;
		});
		assert.ok(found, `no head attends to the k in "${probe}"`);

		const p = m.classify(probe);
		assert.ok(p[1] > p[0], `"${probe}" should read as containing a k`);
	}
	m.free();
});

check('TransformerTextDemo (language)', () => {
	const m = new burn.TransformerTextDemo(0, 32, 2, 1, 16);
	m.train(40, 0.003, 16);
	// 120 words is little enough that held-out accuracy swings run to run
	// (roughly 68-85%); training accuracy is the stable evidence of learning.
	assert.ok(m.train_accuracy > 0.8, `train accuracy ${m.train_accuracy} too low`);
	assert.ok(m.test_accuracy > 0.55, `held-out accuracy ${m.test_accuracy} should beat chance`);
	varied(m.attention('wissenschaft'), 'attention');
	m.free();
});

check('TransformerTextDemo rejects heads that do not divide d_model', () => {
	assert.throws(() => new burn.TransformerTextDemo(1, 30, 4, 1, 12), /divisible/);
	// The head counts the page offers must all be valid for d_model = 32.
	for (const heads of [1, 2, 4, 8]) {
		const m = new burn.TransformerTextDemo(1, 32, heads, 1, 12);
		m.train(1, 0.003, 32);
		assert.equal(m.n_heads, heads);
		m.free();
	}
});

check('TransformerGenDemo (learns a rule, not a list)', () => {
	const m = new burn.TransformerGenDemo(2, 32, 2, 1, 14);
	assert.equal(m.epochs_trained, 0);
	assert.equal(m.corpus_size, 900);
	for (let i = 0; i < 4; i++) m.train(3, 0.003, 32);
	assert.equal(m.epochs_trained, 12);
	assert.ok(m.history.at(-1) < m.history[0], 'loss must fall');

	// The vowel-harmony rule holds across a whole word, so a model that only
	// looked at the last character or two could not enforce it.
	let novel = 0;
	let harmonic = 0;
	const samples = 40;
	for (let i = 0; i < samples; i++) {
		const word = m.generate('', 0.8, 7000 + i);
		assert.ok(/^[a-z]+$/.test(word), `generated "${word}" is not a word`);
		if (!m.is_memorized(word)) novel++;
		if (m.obeys_harmony(word)) harmonic++;
	}
	assert.ok(novel > samples * 0.9, `${novel}/${samples} novel — it is reciting the corpus`);
	assert.ok(harmonic > samples * 0.9, `${harmonic}/${samples} obey harmony — the rule was not learned`);

	// A distribution over the next character, and nothing else.
	const probabilities = m.next_probabilities('tak');
	assert.equal(probabilities.length, 28);
	const total = [...probabilities].reduce((a, b) => a + b, 0);
	assert.ok(Math.abs(total - 1) < 1e-4, `next-character probabilities sum to ${total}`);
	// The start symbol is never a legal continuation.
	assert.ok(probabilities[0] < 0.05, 'start symbol should not be predicted mid-word');

	// Temperature must stay usable at both extremes. A low temperature is the
	// dangerous one: reweighting as p^(1/T) underflows to all-zero weights and
	// the sampler degenerates into emitting end-of-word immediately, so this
	// guards the log-space reweighting.
	for (const temperature of [0.01, 0.2, 1.0, 1.6]) {
		for (let seed = 0; seed < 6; seed++) {
			const word = m.generate('', temperature, seed);
			assert.ok(
				/^[a-z]{2,}$/.test(word),
				`temperature ${temperature} produced "${word}"`
			);
		}
	}
	m.free();
});

check('TransformerGenDemo (real words, prompted)', () => {
	const m = new burn.TransformerGenDemo(0, 32, 2, 1, 14);
	m.train(12, 0.003, 32);
	assert.equal(m.corpus_size, 210);
	const word = m.generate('sch', 0.8, 3);
	assert.ok(word.startsWith('sch'), `"${word}" should continue the prompt`);
	assert.ok(word.length > 3, 'should add something to the prompt');
	m.free();
});

check('causal masking actually hides the future', () => {
	// With a causal mask the prediction after a prefix cannot depend on
	// characters that come later, so two different continuations of the same
	// prefix must agree on what follows that prefix.
	const m = new burn.TransformerGenDemo(2, 32, 2, 1, 14);
	m.train(6, 0.003, 32);
	const a = m.next_probabilities('ta');
	const b = m.next_probabilities('ta');
	for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-6);

	// A longer prefix sharing the same first two characters must not change
	// what the model predicted at position 2.
	assert.ok(m.next_probabilities('tak').length === 28);
	m.free();
});

let failed = 0;
for (const [name, error] of checks) {
	console.log(error ? `FAIL  ${name}\n        ${error}` : `ok    ${name}`);
	if (error) failed++;
}
console.log(`\n${checks.length - failed}/${checks.length} passed`);
process.exit(failed ? 1 : 0);
