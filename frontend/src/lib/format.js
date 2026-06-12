export function formatRetrievalModelLabel(modelName, modelVersion) {
  const name = String(modelName || '').trim();
  const version = String(modelVersion || '').trim().toLowerCase();
  if (!name) {
    return '';
  }
  if (!version || version === 'open_clip') {
    return name;
  }
  if (version === 'open_clip_place') {
    return `${name} place`;
  }
  return `${name} ${modelVersion}`;
}

export function formatMetric(value, digits = 3) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : '0.000';
}

export function formatCoordinate(value, digits = 5) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : 'n/a';
}

export function createClientRetrievalId() {
  if (
    typeof window !== 'undefined' &&
    window.crypto &&
    typeof window.crypto.randomUUID === 'function'
  ) {
    return `locate-${window.crypto.randomUUID()}`;
  }
  return `locate-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function interpolateColor(c1, c2, ratio) {
  const hex = (s) => parseInt(s, 16);
  const r = Math.round(hex(c1.slice(1, 3)) + (hex(c2.slice(1, 3)) - hex(c1.slice(1, 3))) * ratio);
  const g = Math.round(hex(c1.slice(3, 5)) + (hex(c2.slice(3, 5)) - hex(c1.slice(3, 5))) * ratio);
  const b = Math.round(hex(c1.slice(5, 7)) + (hex(c2.slice(5, 7)) - hex(c1.slice(5, 7))) * ratio);
  return `rgb(${r},${g},${b})`;
}

export function buildStreetViewUrl(result) {
  const lat = Number(result?.lat);
  const lon = Number(result?.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return '';
  }
  const params = new URLSearchParams();
  params.set('api', '1');
  params.set('map_action', 'pano');
  params.set('viewpoint', `${lat},${lon}`);
  const heading = Number(result?.heading);
  if (Number.isFinite(heading)) {
    params.set('heading', String(heading));
  }
  const panoId = String(result?.pano_id || '').trim();
  if (panoId) {
    params.set('pano', panoId);
  }
  return `https://www.google.com/maps/@?${params.toString()}`;
}
