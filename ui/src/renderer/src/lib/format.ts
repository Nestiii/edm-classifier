// Display helpers shared across screens.

// Filesystem-safe folder name per subgenre (mirrors config.SUBGENRE_DIRNAMES).
const DIRNAMES: Record<string, string> = {
  'deep house': 'deep_house',
  'tech house': 'tech_house',
  'melodic techno': 'melodic_techno',
  progressive: 'progressive',
  'techno peak time': 'techno_peak_time',
  'hard techno': 'hard_techno',
  'minimal/deep tech': 'minimal_deep_tech',
  trance: 'trance'
}

export function dirnameFor(subgenre: string): string {
  return DIRNAMES[subgenre] ?? subgenre.replace(/[^a-z0-9]+/gi, '_')
}

export function pct(value: number): string {
  return `${Math.round(value * 100)}%`
}

export function basename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path
}

export function seconds(value: number | null): string {
  if (value == null) return '—'
  return value < 10 ? `${value.toFixed(1)} s` : `${Math.round(value)} s`
}
