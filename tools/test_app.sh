#!/usr/bin/env bash
# tools/test_app.sh: end-to-end test of the web app without a browser.
# Starts app.py on port 7861, sends one prompt through the Gradio API, prints the log the page
# would show, then stops the app. Run from anywhere: ./tools/test_app.sh  (or ./demo.sh test-app)
cd "$(dirname "$0")/.." || exit 1
mkdir -p results
t0=$(date +%s)
python -u app.py --port 7861 > results/test_app.log 2>&1 &
APP=$!
for _ in $(seq 1 150); do
  grep -q "Running on local URL" results/test_app.log && break
  kill -0 $APP 2>/dev/null || { echo "app.py exited early:"; tail -20 results/test_app.log; exit 1; }
  sleep 2
done
echo "app ready after $(( $(date +%s) - t0 )) s"
python - <<'PY'
import time
from gradio_client import Client
c = Client('http://127.0.0.1:7861/', verbose=False)
t = time.time()
out = c.predict('A small bench with a flat seat and two legs.', 7, True, api_name='/generate')
print(f'API call took {time.time() - t:.1f} s')
print(out[3])
PY
status=$?
kill $APP; wait $APP 2>/dev/null
exit $status
