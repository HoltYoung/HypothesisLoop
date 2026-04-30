#!/bin/bash
# Run the Build 4 Kimi K2.6 sweep on a temp Compute Engine VM.
# Output streams to /var/log/sweep.log; LLM traces flow to Langfuse Cloud.
# VM auto-deletes via --max-run-duration=40m / --instance-termination-action=DELETE.
set -e
exec > /var/log/sweep.log 2>&1
echo "=== startup: $(date -u) ==="

# 1. System deps
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-pip python3-venv git curl

# 2. Clone repo (public)
cd /root
git clone https://github.com/HoltYoung/HypothesisLoop.git
cd HypothesisLoop

# 3. Pull secrets from instance metadata and write .env
META=http://metadata.google.internal/computeMetadata/v1/instance/attributes
HDR='Metadata-Flavor: Google'
cat > .env <<ENVEOF
OPENAI_API_KEY=$(curl -s -H "$HDR" "$META/openai_key")
KIMI_API_KEY=$(curl -s -H "$HDR" "$META/kimi_key")
MOONSHOT_BASE_URL=https://api.moonshot.ai/v1
DEFAULT_CHAT_PROVIDER=moonshot
DEFAULT_CHAT_MODEL_OPENAI=gpt-4o-mini
DEFAULT_CHAT_MODEL_MOONSHOT=kimi-k2.6
LANGFUSE_PUBLIC_KEY=$(curl -s -H "$HDR" "$META/lf_public")
LANGFUSE_SECRET_KEY=$(curl -s -H "$HDR" "$META/lf_secret")
LANGFUSE_HOST=https://cloud.langfuse.com
ENVEOF

# 4. Python deps in venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the 8-prompt Kimi sweep
echo "=== sweep begin: $(date -u) ==="
printf "ask compute pearson correlations between numeric columns\ny\ny\nask make a histogram of hours_per_week\ny\ny\nask write code that bins age into 5 quantiles and shows income rate per quantile\ny\ny\nask according to the knowledge base when should I use multiple regression?\ny\ny\nask use the knowledge base to recommend an analysis and then run it\ny\ny\nask help me analyze this dataset\ny\ny\nask analyze the column nonexistent_variable_xyz\ny\ny\nask how do I fix my kitchen sink?\ny\ny\nexit\n" | PYTHONPATH=. python builds/build4_rag_router_agent.py --data data/adult.csv --knowledge_dir knowledge --report_dir reports --provider moonshot --session_id a5-sweep-kimi-vm --tags build4,assignment5,sweep,kimi,vm
echo "=== sweep done: $(date -u) ==="

# 6. Best-effort copy stdout off the box (gcloud is preinstalled on standard images).
# If this fails, traces still live in Langfuse Cloud.
NAME=$(curl -s -H "$HDR" http://metadata.google.internal/computeMetadata/v1/instance/name)
ZONE=$(curl -s -H "$HDR" http://metadata.google.internal/computeMetadata/v1/instance/zone | awk -F/ '{print $NF}')
echo "instance=$NAME zone=$ZONE"

echo "=== done; max-run-duration will reap the VM ==="
