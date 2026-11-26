import { useState } from 'react';
import { Search, Plus, Trash2, Loader2 } from 'lucide-react';
import { searchAllClues } from '../engine/clues';
import type { AnalysisResult, ClueMatch, EvidenceEntry } from '../engine/types';

interface ClueBoardPanelProps {
  result: AnalysisResult | null;
  evidenceEntries: EvidenceEntry[];
  onAddEvidence: (entry: EvidenceEntry) => void;
  onToggleEvidence: (id: string) => void;
  onRemoveEvidence: (id: string) => void;
  onClearEvidence: () => void;
}

export function ClueBoardPanel({
  result,
  evidenceEntries,
  onAddEvidence,
  onToggleEvidence,
  onRemoveEvidence,
  onClearEvidence,
}: ClueBoardPanelProps) {
  const [query, setQuery] = useState('');
  const [searching, setSearching] = useState(false);
  const [matches, setMatches] = useState<ClueMatch[]>([]);
  const [lookupError, setLookupError] = useState<string | null>(null);

  const runSearch = async (qOverride?: string) => {
    const q = (qOverride ?? query).trim();
    if (!q || !result) return;
    setSearching(true);
    setLookupError(null);
    try {
      const hits = await searchAllClues(q, result.countries, result.countries[0]?.code);
      setMatches(hits);
    } catch (err) {
      setLookupError(err instanceof Error ? err.message : String(err));
    } finally {
      setSearching(false);
    }
  };

  const applyMatch = (m: ClueMatch) => {
    onAddEvidence({
      id: `${m.id}-${Date.now()}`,
      kind: m.kind,
      label: m.title,
      detail: m.subtitle,
      status: m.status,
      countryCodes: m.countryCodes,
      likelihoodRatio: m.likelihoodRatio,
      active: true,
      source: 'lookup',
    });
  };

  return (
    <div className="subpanel-stack">
      <p className="subpanel-desc">
        Search offline <code>geo-sleuth</code> tables (281 calling codes, 241 driving sides, 60 territories, and 8 regional plate/area-code packs). Applied clues update country probabilities via clamped Bayesian likelihood ratios without discarding competing options.
      </p>

      <div className="preset-row">
        {[
          { label: 'Left-hand traffic', q: 'left' },
          { label: 'Right-hand traffic', q: 'right' },
          { label: '+81 Japan', q: '+81' },
          { label: '+594 Guiana', q: '+594' },
          { label: '010 Beijing', q: '010' },
          { label: 'Plate 06 Ankara', q: '06' },
        ].map((chip) => (
          <button
            key={chip.label}
            type="button"
            className="btn btn-sm btn-pill"
            onClick={() => {
              setQuery(chip.q);
              void runSearch(chip.q);
            }}
          >
            {chip.label}
          </button>
        ))}
      </div>

      <form
        className="inline-search-row"
        onSubmit={(e) => {
          e.preventDefault();
          void runSearch();
        }}
      >
        <input
          type="text"
          className="text-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter calling code (+81), left/right, plate prefix (WP, 京A), or territory…"
          aria-label="Clue lookup query"
        />
        <button
          type="submit"
          className="btn btn-sm btn-primary"
          disabled={!result || !query.trim() || searching}
        >
          {searching ? <Loader2 size={13} aria-hidden="true" /> : <Search size={13} aria-hidden="true" />}
          <span>Lookup</span>
        </button>
      </form>

      {lookupError && <div className="vlm-error">{lookupError}</div>}

      {matches.length > 0 && (
        <div className="clue-results-list">
          {matches.map((m) => (
            <div key={m.id} className="clue-match-row">
              <div style={{ minWidth: 0 }}>
                <div className="lead-name">{m.title}</div>
                <div className="lead-sub">{m.subtitle}</div>
              </div>
              <button
                type="button"
                className="btn btn-sm"
                disabled={m.countryCodes.length === 0}
                title={
                  m.countryCodes.length > 0
                    ? `Boost ${m.countryCodes.slice(0, 6).join(', ')} (${m.likelihoodRatio.toFixed(1)}×)`
                    : 'No matching catalog country codes'
                }
                onClick={() => applyMatch(m)}
              >
                <Plus size={13} aria-hidden="true" />
                <span>Apply ({m.likelihoodRatio.toFixed(1)}×)</span>
              </button>
            </div>
          ))}
        </div>
      )}

      {evidenceEntries.length > 0 && (
        <div className="evidence-ledger">
          <div className="geo-col-head">
            <span className="geo-col-title">
              Active Evidence Board ({evidenceEntries.filter((e) => e.active).length})
            </span>
            <button type="button" className="btn btn-sm btn-ghost" onClick={onClearEvidence}>
              Clear all
            </button>
          </div>
          {evidenceEntries.map((ev) => (
            <div key={ev.id} className={`evidence-row ${ev.active ? '' : 'muted'}`}>
              <label className="evidence-check">
                <input
                  type="checkbox"
                  checked={ev.active}
                  onChange={() => onToggleEvidence(ev.id)}
                />
                <span className={`status-tag status-${ev.status}`}>{ev.status}</span>
                <span className="evidence-label">{ev.label}</span>
              </label>
              <button
                type="button"
                className="btn btn-sm btn-ghost"
                aria-label={`Remove ${ev.label}`}
                onClick={() => onRemoveEvidence(ev.id)}
              >
                <Trash2 size={12} aria-hidden="true" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
