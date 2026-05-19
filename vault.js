import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const execFileAsync = promisify(execFile);

const VAULT_ROOT = path.join(os.homedir(), 'Documents', 'Obsidian Vaults', 'Merced A11Y');
const QMD_PREFIX = 'qmd://merced-a11y/';

function resolveVaultPath(fileUrl) {
  if (!fileUrl?.startsWith(QMD_PREFIX)) return null;
  return path.join(VAULT_ROOT, fileUrl.slice(QMD_PREFIX.length));
}

function parseFrontMatter(content) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return {};
  const result = {};
  for (const line of match[1].split('\n')) {
    const m = line.match(/^([\w-]+):\s*(.+)$/);
    if (!m) continue;
    const [, key, rawVal] = m;
    const val = rawVal.trim();
    if (val.startsWith('[') && val.endsWith(']')) {
      result[key] = val.slice(1, -1).split(',').map(s => s.trim().replace(/^['"]|['"]$/g, ''));
    } else {
      result[key] = val.replace(/^['"]|['"]$/g, '');
    }
  }
  return result;
}

function readVaultFile(vaultAbsPath) {
  try {
    return fs.readFileSync(vaultAbsPath, 'utf8');
  } catch {
    return null;
  }
}

function fieldMatches(frontMatter, field, value) {
  const fv = frontMatter[field];
  if (fv === undefined || fv === null) return false;
  if (Array.isArray(fv)) return fv.some(v => String(v).toLowerCase() === value.toLowerCase());
  return String(fv).toLowerCase() === value.toLowerCase();
}

function applyFilters(hits, filters) {
  if (!filters) return hits;
  const { include = [], exclude = [] } = filters;
  return hits.filter(hit => {
    const fm = hit.frontMatter || {};
    if (exclude.length && exclude.some(r => fieldMatches(fm, r.field, r.value))) return false;
    if (include.length && !include.some(r => fieldMatches(fm, r.field, r.value))) return false;
    return true;
  });
}

export async function searchVault(query, opts = {}) {
  const { n = 5, minScore = 0.4, collection = 'merced-a11y', filters } = opts;
  try {
    const { stdout } = await execFileAsync('qmd', ['query', query, '--json', '-n', String(n), '-c', collection], {
      timeout: 15000,
    });
    const raw = JSON.parse(stdout);
    const hits = raw
      .filter(h => (h.score ?? 0) >= minScore)
      .map(h => {
        const relPath = h.file?.startsWith(QMD_PREFIX) ? h.file.slice(QMD_PREFIX.length) : (h.file || '');
        const absPath = resolveVaultPath(h.file);
        const content = absPath ? readVaultFile(absPath) : null;
        const frontMatter = content ? parseFrontMatter(content) : {};
        return {
          title: h.title || relPath,
          file: h.file || '',
          path: relPath,
          snippet: h.snippet || '',
          score: h.score ?? 0,
          frontMatter,
        };
      });
    return { hits: applyFilters(hits, filters) };
  } catch {
    return { hits: [] };
  }
}
