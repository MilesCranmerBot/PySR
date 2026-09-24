export function menuVersions(versions) {
  const parsed = versions.map(version => {
    const match = /^v(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:(a|b|rc)(\d+))?$/.exec(version);
    if (!match || (match[4] && match[3] === undefined)) return null;
    const [, major, minor, patch, prerelease, prereleaseNumber] = match;
    return {
      major,
      key: minor === undefined ? major : `${major}.${minor}`,
      rank: [patch === undefined ? -1 : Number(patch),
        prerelease === undefined ? 3 : { a: 0, b: 1, rc: 2 }[prerelease],
        Number(prereleaseNumber ?? 0)],
    };
  });
  const best = new Map();
  for (const entry of parsed) {
    if (!entry) continue;
    const previous = best.get(entry.key);
    if (!previous) {
      best.set(entry.key, entry);
      continue;
    }
    for (let i = 0; i < entry.rank.length; i++) {
      if (entry.rank[i] === previous.rank[i]) continue;
      if (entry.rank[i] > previous.rank[i]) best.set(entry.key, entry);
      break;
    }
  }
  return versions.filter((version, index) => {
    if (version === 'dev' || version === 'stable') return true;
    const entry = parsed[index];
    return entry && best.get(entry.key) === entry &&
      (entry.key !== entry.major || ![...best.keys()].some(key => key.startsWith(`${entry.major}.`)));
  });
}
