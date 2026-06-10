async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body?.detail || 'Request failed');
  }
  return body;
}

export function startLocatorEval(payload) {
  return requestJson('/api/eval/locator/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}

export function getLocatorEvalStatus(evalId) {
  return requestJson(`/api/eval/locator/status/${encodeURIComponent(evalId)}`, {
    cache: 'no-store'
  });
}

export function stopLocatorEval(evalId) {
  return requestJson(`/api/eval/locator/stop/${encodeURIComponent(evalId)}`, {
    method: 'POST'
  });
}

export function startLocatorEvalDatasetScrape(payload) {
  return requestJson('/api/eval/locator/dataset/scrape', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
}

export function getLocatorEvalDatasetStatus(datasetId) {
  return requestJson(
    `/api/eval/locator/dataset/status/${encodeURIComponent(datasetId)}`,
    { cache: 'no-store' }
  );
}

export function stopLocatorEvalDataset(datasetId) {
  return requestJson(
    `/api/eval/locator/dataset/stop/${encodeURIComponent(datasetId)}`,
    { method: 'POST' }
  );
}
