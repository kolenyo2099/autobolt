import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const srcDir = path.join(rootDir, 'node_modules', 'onnxruntime-web', 'dist');
const destDir = path.join(rootDir, 'public', 'ort');

if (!fs.existsSync(srcDir)) {
  console.warn('[copy-ort-assets] onnxruntime-web/dist not found yet, skipping.');
  process.exit(0);
}

fs.mkdirSync(destDir, { recursive: true });

const entries = fs.readdirSync(srcDir);
let copied = 0;
for (const entry of entries) {
  if (/^ort-wasm.*\.(wasm|mjs)$/.test(entry)) {
    fs.copyFileSync(path.join(srcDir, entry), path.join(destDir, entry));
    copied++;
  }
}

console.log(`[copy-ort-assets] Copied ${copied} ONNX Runtime Web assets to public/ort/`);
