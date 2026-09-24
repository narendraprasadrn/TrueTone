const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: "new", args: ['--no-sandbox'] });
  try {
    const page = await browser.newPage();
    page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
    
    await page.goto('http://localhost:3005/test-bench', { waitUntil: 'networkidle0' });
    
    const html = await page.content();
    if (html.includes('VOICE AUTHENTICITY RESULT')) {
      console.log("Page loaded successfully.");
    }
  } catch(e) {
    console.error(e);
  } finally {
    await browser.close();
  }
})();
