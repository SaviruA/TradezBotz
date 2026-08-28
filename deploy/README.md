# Deploying to a GCP always-free VM

Everything here targets the `e2-micro` always-free tier. It stays free only
inside these constraints:

- **Region must be `us-west1`, `us-central1`, or `us-east1`.** Any other region
  bills at standard rates.
- **One `e2-micro` instance**, 30 GB standard persistent disk, 1 GB egress/month.
  Our workload downloads far more than it uploads, so egress is not a concern.

GCP's free tier still requires a billing account on file, and it is possible to
provision something outside the free tier by accident. **Set a budget alert at
$1 before doing anything else.**

## 1. Create the instance

```bash
gcloud compute instances create tradezbotz \
  --project=YOUR_PROJECT_ID \
  --zone=us-east1-b \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-standard
```

```bash
gcloud compute ssh tradezbotz --zone=us-east1-b
```

## 2. Get the code onto the box

The repo is private, so plain `git clone` over HTTPS will prompt for
credentials. Two options — a read-only **deploy key** is the safer one, since it
grants access to this repo alone rather than the whole account:

```bash
ssh-keygen -t ed25519 -C "tradezbotz-vm" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

Add that public key at **repo → Settings → Deploy keys → Add deploy key**
(read-only). Then:

```bash
git clone git@github.com:SaviruA/TradezBotz.git ~/tradezbotz
```

## 3. Configure secrets

`.env` is gitignored and therefore not in the clone. Create it on the box:

```bash
cd ~/tradezbotz && cp .env.example .env && nano .env
```

Fill in `SEC_USER_AGENT` (must contain a real email — the SEC blocks requests
without one) and `MASSIVE_API_KEY`. `setup_vm.sh` refuses to proceed while the
placeholder address is still in place.

## 4. Run setup

```bash
bash ~/tradezbotz/deploy/setup_vm.sh ~/tradezbotz
```

Installs Python and dependencies, creates the venv, installs three systemd
units, and enables the daily ingest timer.

## 5. Kick off the historical work

```bash
cd ~/tradezbotz
./.venv/bin/python -m tradezbotz ingest-edgar --days 730
./.venv/bin/python -m tradezbotz enqueue-symbols
sudo systemctl start tradezbotz-backfill
```

Watch it, then disconnect freely — systemd owns the process, not your SSH
session:

```bash
journalctl -u tradezbotz-backfill -f
```

## What runs on a schedule

| Unit | When | What |
|---|---|---|
| `tradezbotz-ingest.timer` | daily 23:30 ET | pull the last 5 days of Form 4 filings, queue new symbols |
| `tradezbotz-backfill.service` | manual / on-failure restart | work the symbol queue at 5 req/min |

23:30 ET is deliberate: it sits after the 22:00 ET Form 4 dissemination cutoff,
so a day's filings are complete when we pull them. The 5-day lookback means a
missed run self-heals — ingestion is idempotent by accession number.

## Operating it

```bash
./.venv/bin/python -m tradezbotz status          # events, cached symbols, queue
systemctl list-timers tradezbotz-ingest.timer    # is the daily job scheduled
journalctl -u tradezbotz-ingest --since today    # what did it do
sudo systemctl stop tradezbotz-backfill          # clean stop at symbol boundary
```

Stopping is safe at any moment: the runner checkpoints after every symbol and
handles `SIGTERM` by finishing the current one before exiting.

## Backups

The whole pipeline state is three SQLite files in `state/`. Back them up before
anything destructive:

```bash
sqlite3 state/events.db ".backup state/events-$(date +%F).db"
tar czf ~/tradezbotz-state-$(date +%F).tar.gz state/
```

`state/` is gitignored, so it is never pushed. The event store is the only
irreplaceable artifact — bars can always be refetched, but a point-in-time
record of what was knowable when cannot be reconstructed after the fact.
