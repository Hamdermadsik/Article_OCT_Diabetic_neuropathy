// Run with Node.js and Playwright installed. No real review files are accessed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('playwright');

const html = fs.readFileSync(path.join(__dirname, '../src/qc_review.html'), 'utf8')
  .replace('/*APP_CONFIG*/', JSON.stringify({token: 'test', showNames: false, warnings: []}));
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64');

function fixture() {
  const items = Array.from({length: 50}, (_, i) => ({
    id: `scan-${i}`, filename: `scan-${i}.dcm`, width: 1355, height: 1355, bscan: null
  }));
  const reviews = Object.fromEntries(items.map(item => [item.id, {
    filename: item.filename, status: 'discard', reasons: [], note: '',
    reviewed_at: '2026-01-01T00:00:00Z'
  }]));
  for (const i of [1, 4, 6]) reviews[`scan-${i}`].status = 'use';
  for (const i of [2, 5]) reviews[`scan-${i}`].status = 'unsure';
  for (const i of [3, 7]) delete reviews[`scan-${i}`];
  for (const i of [1, 2, 10, 30]) reviews[`scan-${i}`].reasons = ['segmentation'];
  return {items, reviews, revision: 0, results_path: 'mock-reviews.json'};
}

async function ready(page, id) {
  await page.waitForFunction(id => document.getElementById('name').textContent === `${id}.dcm`
    && !document.querySelector('[data-status="use"]').disabled, id);
}

async function selectScan(page, index) {
  await page.getByRole('button', {name: `Scan ${String(index + 1).padStart(4, '0')}`, exact: false}).click();
  await ready(page, `scan-${index}`);
}

(async () => {
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined});
  async function scenario(name, test) {
    const context = await browser.newContext({viewport: {width: 1456, height: 849}});
    const page = await context.newPage();
    const state = fixture(), posts = [], errors = [];
    let failSave = false;
    page.on('pageerror', error => errors.push(error.message));
    await page.route('http://qc.test/**', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/') return route.fulfill({contentType: 'text/html', body: html});
      if (url.pathname === '/api/state') return route.fulfill({json: state});
      if (url.pathname.startsWith('/api/image/')) return route.fulfill({contentType: 'image/png', body: png});
      if (url.pathname === '/api/review') {
        const payload = route.request().postDataJSON();
        posts.push(payload);
        if (failSave) return route.fulfill({status: 409, json: {error: 'Mock save rejected'}});
        assert.equal(payload.revision, state.revision);
        state.revision++;
        const review = {...payload, filename: `${payload.id}.dcm`, reviewed_at: '2026-01-02T00:00:00Z'};
        state.reviews[payload.id] = review;
        return route.fulfill({json: {revision: state.revision, review, warning: null}});
      }
      throw Error(`Unexpected request: ${url.pathname}`);
    });
    try {
      await page.goto('http://qc.test/');
      await ready(page, 'scan-3');
      await test({page, state, posts, rejectSave: () => {failSave = true;}});
      assert.deepEqual(errors, []);
      console.log(`PASS ${name}`);
    } catch (error) {
      console.error(await page.evaluate(() => {
        const list = document.getElementById('list');
        return {selected: document.getElementById('name').textContent,
          scrollTop: list.scrollTop, height: list.clientHeight, scrollHeight: list.scrollHeight,
          active: list.querySelector('[aria-current="true"]')?.getBoundingClientRect().toJSON()};
      }));
      throw error;
    } finally {
      // Also permits closing deliberately dirty test forms without saving.
      page.on('dialog', dialog => dialog.accept());
      await context.close();
    }
  }
  try {
    await scenario('discard queue advances in order and retains sidebar position', async ({page, posts}) => {
      await page.selectOption('#filter', 'discard');
      await selectScan(page, 35);
      const top = await page.locator('#list').evaluate(list => list.scrollTop);
      assert(top > 0);
      await page.evaluate(() => renderList());
      assert.equal(await page.locator('#list').evaluate(list => list.scrollTop), top);
      await page.click('[data-status="use"]');
      await ready(page, 'scan-36');
      assert.equal(await page.locator('#filter').inputValue(), 'discard');
      assert(await page.locator('#list').evaluate(list => list.scrollTop) > 0);
      await page.click('[data-status="discard"]');
      await ready(page, 'scan-37');
      assert.deepEqual(posts.map(p => p.id), ['scan-35', 'scan-36']);
    });
    await scenario('maybe and use queues advance to already-reviewed scans', async ({page}) => {
      await page.selectOption('#filter', 'unsure');
      await ready(page, 'scan-2');
      await page.click('[data-status="discard"]');
      await ready(page, 'scan-5');
      await page.selectOption('#filter', 'use');
      await ready(page, 'scan-1');
      await page.click('[data-status="unsure"]');
      await ready(page, 'scan-4');
    });
    await scenario('DEJ filter is independent of decision and uses saved flags', async ({page, state}) => {
      assert.equal(await page.locator('option[value="segmentation"]').textContent(), 'Forkert DEJ');
      await page.selectOption('#filter', 'segmentation');
      await ready(page, 'scan-1');
      assert.equal(await page.locator('#list .scan').count(), 4);
      assert.deepEqual([...new Set([1, 2, 10, 30].map(i => state.reviews[`scan-${i}`].status))].sort(),
        ['discard', 'unsure', 'use']);
      await page.click('[data-status="use"]');
      await ready(page, 'scan-2');
      assert.equal(await page.locator('#list .scan').count(), 4);
      await page.uncheck('input[value="segmentation"]');
      assert.equal(await page.locator('#list .scan').count(), 4);
      await page.click('[data-status="unsure"]');
      await ready(page, 'scan-10');
      assert.equal(await page.locator('#list .scan').count(), 3);
      assert.deepEqual(state.reviews['scan-2'].reasons, []);
    });
    await scenario('last filtered scan does not wrap to the start', async ({page}) => {
      await page.selectOption('#filter', 'discard');
      await selectScan(page, 49);
      await page.click('[data-status="use"]');
      await ready(page, 'scan-48');
      assert.equal(await page.locator('#next').isDisabled(), true);
    });
    await scenario('empty queue clears selection and can be left safely', async ({page}) => {
      await page.selectOption('#filter', 'unreviewed');
      await ready(page, 'scan-3');
      await page.click('[data-status="use"]');
      await ready(page, 'scan-7');
      await page.click('[data-status="use"]');
      await page.waitForFunction(() => document.querySelector('#list .empty') !== null);
      assert.equal(await page.locator('[data-status="use"]').isDisabled(), true);
      assert.equal(await page.locator('#mip').getAttribute('src'), null);
      await page.selectOption('#filter', 'discard');
      await ready(page, 'scan-0');
      await page.selectOption('#filter', 'unreviewed');
      assert.equal(await page.locator('[data-status="use"]').isDisabled(), true);
    });
    await scenario('all-images queue still prioritizes unreviewed scans', async ({page}) => {
      await page.click('[data-status="use"]');
      await ready(page, 'scan-7');
    });
    await scenario('failed saves keep the current scan and draft', async ({page, rejectSave}) => {
      await page.selectOption('#filter', 'discard');
      await selectScan(page, 20);
      await page.fill('#note', 'Do not lose this draft');
      rejectSave();
      await page.click('[data-status="use"]');
      await page.waitForFunction(() => document.getElementById('error').textContent === 'Mock save rejected');
      await ready(page, 'scan-20');
      assert.equal(await page.locator('#note').inputValue(), 'Do not lose this draft');
      assert.equal(await page.locator('#filter').inputValue(), 'discard');
    });
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
