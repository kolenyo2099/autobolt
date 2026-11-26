import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const distDir = path.join(rootDir, 'dist');
const distOrtDir = path.join(distDir, 'ort');
const distModelsDir = path.join(distDir, 'models');
const chunksDir = path.join(distModelsDir, 'chunks');

const CHUNK_SIZE = 40 * 1024 * 1024; // 40 MiB
const MAX_GITHUB_PAGES_FILE = 48 * 1024 * 1024;
const MAX_TOTAL_BYTES = 950 * 1024 * 1024;

if (!fs.existsSync(distDir)) {
  throw new Error('dist/ does not exist. Run "pnpm build" first.');
}

// 1. Trim unused ONNX Runtime WASM files in dist/ort/
const KEEP_ORT = new Set([
  'ort-wasm-simd-threaded.asyncify.wasm',
  'ort-wasm-simd-threaded.asyncify.mjs',
]);
if (fs.existsSync(distOrtDir)) {
  let removed = 0;
  for (const name of fs.readdirSync(distOrtDir)) {
    if (!KEEP_ORT.has(name)) {
      fs.rmSync(path.join(distOrtDir, name), { force: true });
      removed++;
    }
  }
  console.log(`[prepare-pages-dist] Trimmed ${removed} unused WASM files from dist/ort/`);
}

// 2. Split any ONNX file > 40 MiB into 40 MiB parts + .manifest.json
fs.mkdirSync(chunksDir, { recursive: true });
if (fs.existsSync(distModelsDir)) {
  for (const name of fs.readdirSync(distModelsDir)) {
    if (!name.endsWith('.onnx')) continue;
    const fullPath = path.join(distModelsDir, name);
    const stat = fs.statSync(fullPath);
    if (!stat.isFile() || stat.size <= CHUNK_SIZE) continue;

    const buf = fs.readFileSync(fullPath);
    const parts = [];
    let offset = 0;
    let index = 0;
    while (offset < buf.length) {
      const end = Math.min(offset + CHUNK_SIZE, buf.length);
      const partName = `${name}.part${String(index).padStart(2, '0')}`;
      fs.writeFileSync(path.join(chunksDir, partName), buf.subarray(offset, end));
      parts.push(`models/chunks/${partName}`);
      offset = end;
      index++;
    }

    const manifestPath = path.join(distModelsDir, `${name}.manifest.json`);
    fs.writeFileSync(
      manifestPath,
      JSON.stringify({ totalBytes: buf.length, parts }, null, 2),
    );
    fs.rmSync(fullPath, { force: true });
    console.log(
      `[prepare-pages-dist] Split ${name} (${(buf.length / 1048576).toFixed(1)} MB) into ${parts.length} parts`,
    );
  }
}

// 3. Ensure .nojekyll exists for GitHub Pages
fs.writeFileSync(path.join(distDir, '.nojekyll'), '');

// 4. Verify per-file and total sizes
let totalBytes = 0;
let maxFile = { rel: '', size: 0 };
function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(p);
    } else if (entry.isFile()) {
      const sz = fs.statSync(p).size;
      totalBytes += sz;
      if (sz > maxFile.size) {
        maxFile = { rel: path.relative(distDir, p), size: sz };
      }
      if (sz > MAX_GITHUB_PAGES_FILE) {
        throw new Error(
          `File ${path.relative(distDir, p)} is ${(sz / 1048576).toFixed(2)} MiB (> 48 MiB limit)`,
        );
      }
    }
  }
}
walk(distDir);

if (totalBytes > MAX_TOTAL_BYTES) {
  throw new Error(
    `Total dist size ${(totalBytes / 1048576).toFixed(1)} MiB exceeds ${MAX_TOTAL_BYTES / 1048576} MiB limit`,
  );
}

console.log(
  `[prepare-pages-dist] Verified dist/: ${(totalBytes / 1048576).toFixed(1)} MiB total, largest file ${maxFile.rel} (${(maxFile.size / 1048576).toFixed(1)} MiB)`,
);
