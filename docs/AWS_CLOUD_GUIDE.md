# Running the Experiments on AWS — Beginner Guide

You have never used cloud before. This guide assumes **zero** prior knowledge.
It walks you from "no account" to "results downloaded, billing stopped", with
every term explained the first time it appears.

**What you are doing, in one sentence:** renting a powerful computer from Amazon
by the hour, copying this project onto it, running the training, downloading the
results, then giving the computer back so you stop paying.

---

## 0. The mental model (read this first)

- **AWS** (Amazon Web Services) = Amazon rents out computers over the internet.
- **EC2** = the AWS service that rents you a virtual computer. One rented
  computer is called an **instance**.
- You pay **per second the instance is running**. Turn it off → stop paying for
  compute. This is the single most important thing in this guide: **when you are
  done, you must STOP or TERMINATE the instance or you keep getting charged.**
- The instance is a normal Linux machine. You connect to it with **SSH** (a
  secure remote terminal) from your Mac's Terminal. Once connected it feels
  exactly like a terminal on your own laptop.

Why do this at all? This project's full run is ~11 days on your laptop serially,
or ~1 day on a 16-core cloud machine using `run_parallel.sh`. See
[the cost table](#11-cost-cheat-sheet) — the whole thing costs a few dollars.

---

## 1. Cost expectations (so there are no surprises)

The machine we recommend is a **c7i.4xlarge**: 16 vCPUs (virtual CPU cores),
32 GB RAM. No GPU — this project does not use one.

| Pricing type | Price/hour (approx, us-east-1) | Full run (~24h) |
|--------------|-------------------------------|-----------------|
| **On-demand** (guaranteed, never interrupted) | ~$0.71 | **~$17** |
| **Spot** (up to 90% off, but AWS can reclaim it) | ~$0.20–0.30 | **~$5–7** |

Plus tiny extras: disk storage ~$0.10 for a day, data download ~$0.10. Total
still under $20 on-demand, under $10 spot.

> **Recommendation for a first-timer: use On-Demand.** Spot is cheaper but AWS
> can take the machine back mid-run (our script survives this — see §8 — but it
> adds confusion on your first try). Do your first run On-Demand, switch to Spot
> once comfortable.

Set up a **billing alarm** (§10) so you get emailed if spending crosses a limit.

> **Reality check on `MODE=full` timing (measured, not estimated).** The
> hyperparameter **tuning** phase is the slow part and it does **not** use all 16
> cores — it runs one Optuna study per algorithm (4 jobs), each doing 30 trials
> sequentially. On a c7i.4xlarge a trial measured **~70 min** (`~4250 s/it`), so
> **tuning alone takes ~35 h** (all 4 algos run in parallel at a similar rate,
> so wall-clock ≈ one algo's ~35 h, not 4×). The 16-core parallelism only kicks
> in **after** tuning, during the train/eval phase. So budget **~1.5–2 days**
> wall-clock for a real `full` run, not the ~24 h first implied — and on-demand
> that is **~$25–35**, not ~$17. To see progress:
> ```bash
> # the % and trial count are on the last progress-bar line:
> grep -oE '[0-9]+/30 \[[^]]*\]' logs/parallel_*/tune_dqn.log | tail -1
> ```
> If that's too slow/expensive, use `MODE=overnight` (fewer trials + shorter
> steps → results in ~2–3 h) for a cheap end-to-end pass, and only pay for
> `full` when you actually need publication-size numbers.

---

## 2. Create an AWS account (~15 min, one time)

1. Go to <https://aws.amazon.com/> → **Create an AWS Account**.
2. Enter email, pick an account name (e.g. "sudwipto-personal").
3. **You must enter a credit/debit card.** AWS bills to it. New accounts get a
   "Free Tier" but the machine we use is **not** free-tier — expect the small
   charges in §1.
4. Verify your phone number.
5. Choose the **Basic support plan** (free).
6. Sign in to the **AWS Management Console** (the web dashboard):
   <https://console.aws.amazon.com/>.

### Pick a Region and never touch it again
Top-right of the console shows a **Region** (e.g. "N. Virginia us-east-1").
A Region is the physical datacenter location. Everything you create lives in one
Region and is invisible from others — so **pick one and stay in it** the whole
time. `us-east-1` (N. Virginia) is the cheapest and has the most capacity; use
it. Write down which one you chose.

---

## 3. Key concepts you'll meet in the launch screen

You do not need to master these — just recognize them when they appear:

- **AMI** (Amazon Machine Image) = the operating system template the instance
  boots from. We use **Ubuntu 24.04 LTS** (a common, well-documented Linux).
- **Instance type** = the hardware size. We use **c7i.4xlarge** (the "c" family
  is compute-optimized = lots of CPU, which is exactly what SUMO needs).
- **Key pair** = a cryptographic file (`.pem`) that is your password to SSH in.
  AWS gives you the file **once** at creation — lose it and you can't log in.
- **Security group** = a firewall. By default it blocks everything; you open
  just **port 22** (SSH) so you can connect.
- **EBS volume** = the instance's hard disk. Default ~8 GB is too small; we set
  **30 GB**.

---

## 4. Create your SSH key pair (one time)

1. Console search bar → type **EC2** → open the EC2 service.
2. Left menu → **Network & Security → Key Pairs** → **Create key pair**.
3. Name: `traffic-key`. Type: **RSA**. Format: **.pem**. → **Create**.
4. Your browser downloads `traffic-key.pem`. **This is your only copy.** Move it
   somewhere safe and lock down its permissions (SSH refuses loose keys):

   ```bash
   mkdir -p ~/.ssh
   mv ~/Downloads/traffic-key.pem ~/.ssh/
   chmod 400 ~/.ssh/traffic-key.pem
   ```

---

## 5. Launch the instance (the rented computer)

1. EC2 console → big orange **Launch instance** button.
2. **Name**: `traffic-training`.
3. **Application and OS Image (AMI)**: search **Ubuntu**, pick
   **Ubuntu Server 24.04 LTS** (64-bit x86, free-tier-eligible label is fine —
   the AMI is free, the instance size is what costs).
   - **The Python version does not actually matter** — §7 installs its own
     Python 3.11 with `uv` regardless of what the AMI ships. So if you end up on
     a newer Ubuntu (25.x/26.04) it's fine; just follow §7 as written.
4. **Instance type**: click the dropdown, type `c7i.4xlarge`, select it.
   - Can't find it / "not available"? Use `c6i.4xlarge` (older, near-identical)
     or `m7i.4xlarge`. Any 16-vCPU instance works.
5. **Key pair**: choose `traffic-key` (the one from §4).
6. **Network settings** → **Edit**:
   - Ensure **Allow SSH traffic from** is checked.
   - Set it to **My IP** (safest — only your current internet connection can
     connect). If your IP changes later you'll re-add it; that's fine.
7. **Configure storage**: change the root volume from 8 GB to **30 GB**, type
   **gp3**.
8. *(Optional, to use Spot pricing)* Expand **Advanced details** → **Purchase
   option** → check **Request Spot Instances**. Skip for your first run.
   - New accounts may show **"Max spot instance count exceeded"**. Check
     **Service Quotas → EC2 → All Standard Spot Instance Requests** — if it's
     already ≥ 16 you're fine (16 = the vCPUs of a 4xlarge); no increase needed,
     just retry. If it's 0, either request an increase (minutes–hours) or use
     On-Demand.
   - **You cannot convert a running On-Demand instance to Spot** (or vice-versa).
     The purchase option is fixed at launch — to switch, terminate and relaunch.
9. Review the **Summary** panel on the right, then **Launch instance**.
10. Click the instance ID link → you land on the instances list. Wait until
    **Instance state = Running** and **Status checks = 2/2 passed** (~1–2 min).

---

## 6. Connect to it via SSH

1. In the instances list, click your instance. Copy its **Public IPv4 address**
   (looks like `54.81.x.x`).
2. In your Mac Terminal:

   ```bash
   ssh -i ~/.ssh/traffic-key.pem ubuntu@PUBLIC_IP
   ```

   Replace `PUBLIC_IP` with the address. `ubuntu` is the default username for
   Ubuntu AMIs.
3. First time it asks *"Are you sure you want to continue connecting?"* → type
   `yes`. You're now inside the cloud machine — the prompt changes to something
   like `ubuntu@ip-172-31-x-x:~$`. Everything you type now runs on the rented
   computer.

> **If it hangs / "Connection timed out":** the security group isn't allowing
> your IP. EC2 → your instance → **Security** tab → click the security group →
> **Edit inbound rules** → add rule: Type **SSH**, Source **My IP** → Save.

---

## 7. Set up and run the project (on the instance)

You're now SSH'd in. Run these on the cloud machine.

> **Why we don't use the system Python.** `requirements.txt` was pinned on a Mac
> with Python 3.9. Fresh Ubuntu AMIs ship a much newer Python (24.04 = 3.12,
> 26.04 = 3.14) that has **no matching wheels** for pins like `torch==2.8.0`, so
> `pip install` fails and nothing installs. The fix that works on *any* Ubuntu:
> install an isolated Python 3.11 with **`uv`** (a fast package manager that
> downloads its own standalone Python — no `apt`, no compiling).

```bash
# 7.1 system packages: git + tmux, PLUS the X11 shared libraries the SUMO
# binary links against. A minimal Ubuntu server lacks these, and without them
# every run dies instantly with:
#   sumo: error while loading shared libraries: libXrender.so.1
sudo apt-get update
sudo apt-get install -y git tmux curl \
    libxrender1 libxext6 libxfixes3 libxcursor1 libxrandr2 libxi6 libgl1

# 7.2 get the code (public repo clone; for a private repo see note below)
git clone https://github.com/sudwiptokm/group-project.git group_project
cd group_project

# 7.3 install uv, then put it on PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"       # if `uv` still "not found", also try: source $HOME/.local/bin/env
uv --version                               # confirm it prints a version

# 7.4 build a Python 3.11 venv and install deps (~30 s with uv)
rm -rf venv
uv python install 3.11
uv venv --python 3.11 venv
source venv/bin/activate
uv pip install -r requirements.txt

# 7.5 tell the tools where SUMO lives
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
echo "$SUMO_HOME"                          # should print a path, not an error
```

> If `torch==2.8.0` still shows "No matching distribution" even on 3.11, that
> exact version was pulled from the index — relax just that pin:
> `sed -i 's/^torch==.*/torch>=2.8,<2.10/' requirements.txt` then re-run
> `uv pip install -r requirements.txt`. (SB3 2.7.1 runs fine on torch 2.9.)

> **Private repo?** Easiest is to make the GitHub repo public temporarily, or
> create a GitHub **Personal Access Token** and clone with
> `git clone https://<TOKEN>@github.com/sudwiptokm/group-project.git`.
> Do NOT put your token in any committed file.

### Start the run inside `tmux` (survives disconnects)
`tmux` keeps the job alive even if your laptop sleeps or SSH drops. Without it, a
dropped connection kills a multi-hour run.

```bash
tmux new -s train        # opens a persistent session

# inside tmux, re-set the three env things (a fresh tmux shell doesn't inherit them):
source venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
MODE=full JOBS=16 ./run_parallel.sh
```

- `MODE=full` = the publication-size budget (~1 day). Use `MODE=overnight`
  (~2–3 h) for a quick cheap end-to-end test first.
- `JOBS=16` = run 16 jobs at once (matches the 16 vCPUs).

**Confirm it started:** within ~2 min you should see lines like `[start] tune_dqn`.
If you see an immediate `[FAIL]`, read the named log in `logs/parallel_<stamp>/`.

### Now close your laptop and walk away
1. **Detach** from tmux: press `Ctrl-b`, release, then `d`. The run keeps going
   **on the cloud machine** — it does not depend on your laptop.
2. You can now shut the laptop lid, disconnect wifi, or turn it off. The cloud
   box keeps computing.

> tmux is what makes this safe. Without it, an SSH disconnect (laptop sleep,
> wifi drop) kills the run. Always launch inside `tmux new -s train`.

### Check progress whenever you like
From any machine with your key, laptop back on:
```bash
ssh -i ~/.ssh/traffic-key.pem ubuntu@PUBLIC_IP

# either watch live:
tmux attach -t train
#   ...then detach again with Ctrl-b then d

# or a quick glance without attaching:
tail -n 30 ~/group_project/logs/parallel_*/driver.log
grep -c '\[ok\]'   ~/group_project/logs/parallel_*/driver.log   # jobs finished
grep    '\[FAIL\]' ~/group_project/logs/parallel_*/driver.log   # any failures
```
You'll know it's finished when the driver log ends with `=== COMPARE ===`,
`=== PLOTS ===`, then `done.`.

---

## 8. When it finishes

The script auto-runs the final aggregation, producing:
- `logs/comparison.csv` — the RL-vs-fixed-time results table (mean ± std).
- `results/*.png` — the bar charts and trade-off curves.
- `models/*.zip` — the trained policies (reusable for demos/inference).

### Download the results to your Mac
Run these **on your laptop** (a new Terminal window, NOT the SSH session).
`scp` = secure copy over SSH. Replace `PUBLIC_IP`.

```bash
cd ~/Desktop/group_project     # your local copy

scp -i ~/.ssh/traffic-key.pem -r ubuntu@PUBLIC_IP:~/group_project/results  ./cloud_results
scp -i ~/.ssh/traffic-key.pem -r ubuntu@PUBLIC_IP:~/group_project/logs     ./cloud_logs
scp -i ~/.ssh/traffic-key.pem -r ubuntu@PUBLIC_IP:~/group_project/models   ./cloud_models
scp -i ~/.ssh/traffic-key.pem -r ubuntu@PUBLIC_IP:~/group_project/params   ./cloud_params
```

Now you have everything locally. Open `results/*.png` and `logs/comparison.csv`.

### Spot interruption — important if you ran on Spot
If AWS reclaims a Spot instance, it **terminates** and, by default, its disk is
**deleted** — so any `models/logs/params` not already copied off the box are
lost. The `run_parallel.sh` script *is* resumable (it skips artifacts that
already exist), but only if those artifacts still exist somewhere. On a ~24h
`full` run with your laptop closed, a reclaim mid-run = start over.

Your options, cheapest peace-of-mind first:
- **Accept the risk.** If reclaimed, launch a fresh instance, redo §7, rerun the
  same command. You lose the completed work but it's only compute time. Fine for
  a student run; Spot reclaims of a common instance type are not that frequent.
- **Use On-Demand instead** (§5.8, leave Spot unchecked). Never interrupted;
  costs ~$17 for the full run vs ~$5–7 Spot. Simplest way to safely walk away.
- **Auto-back-up from the box to S3.** Survives reclaim even with your laptop
  off, because the copy runs *on the instance*. Full walkthrough below.

### Protect a Spot run with S3 auto-backup (recommended for `full`)

**S3** = Amazon's file storage. The idea: a small loop on the instance copies
your results to an S3 bucket every 10 minutes. If Spot reclaims the box, the
results are safe in S3 — you relaunch, pull them back, and resume. Storage cost
is pennies. Your training run keeps going through all of this; nothing here
interrupts it.

> Examples below use region **eu-west-2** (London). Use whichever region your
> instance is in — keep it consistent.

**Part A — Create the S3 bucket (browser).**
1. Console search → **S3** → **Create bucket**.
2. **Bucket name**: must be globally unique, lowercase, no spaces — e.g.
   `sudwipto-traffic-backup-2607`. **Write it down.**
3. **Region**: same as your instance (e.g. **EU (London) eu-west-2**).
4. Leave the rest default (Block Public Access on is fine). → **Create bucket**.

**Part B — Create an IAM role and attach it to the instance.**
This lets the instance write to S3 with **no access keys stored on the box**.
1. Console search → **IAM** → **Roles** → **Create role**.
2. **Trusted entity**: **AWS service**. **Use case**: **EC2**. → **Next**.
3. Check **AmazonS3FullAccess**. → **Next**.
4. **Role name**: `ec2-s3-backup` → **Create role**.
5. **EC2 → Instances** → check your instance → **Actions → Security → Modify IAM
   role** → pick **ec2-s3-backup** → **Update IAM role**. Applies live — no
   reboot, run unaffected.

**Part C — On the instance, in a *second* tmux window** (leaves training alone):
```bash
tmux attach -t train
# open a new window: press Ctrl-b then c
```
Then in that new window:
```bash
sudo snap install aws-cli --classic          # install AWS CLI
aws --version

aws sts get-caller-identity                   # proves the role works (prints an ARN, no keys)

export BUCKET=sudwipto-traffic-backup-2607    # <-- your bucket name
export AWS_DEFAULT_REGION=eu-west-2           # <-- your region
aws s3 ls s3://$BUCKET                         # confirm bucket reachable
```
If `get-caller-identity` or `s3 ls` errors, the role probably isn't attached yet
— wait 30 s and retry, or recheck Part B step 5.

**Part D — Start the background backup loop.**
```bash
cat > ~/s3backup.sh <<'EOF'
#!/usr/bin/env bash
BUCKET="${BUCKET:?set BUCKET}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-eu-west-2}"
cd ~/group_project || exit 1
while true; do
  for d in models logs params results; do
    [ -d "$d" ] && aws s3 sync "$d" "s3://$BUCKET/$d" --only-show-errors
  done
  echo "[$(date '+%H:%M:%S')] synced to s3://$BUCKET"
  sleep 600      # every 10 min
done
EOF
chmod +x ~/s3backup.sh

BUCKET=$BUCKET nohup ~/s3backup.sh > ~/s3backup.log 2>&1 &
echo "backup PID $!"
```
Verify after ~10 min, then detach (`Ctrl-b` then `d`) and close the laptop:
```bash
tail ~/s3backup.log
aws s3 ls s3://$BUCKET/logs/
```

**Recovery — if Spot reclaims the instance.**
1. Launch a fresh Spot instance; redo §7 setup (uv/venv/deps).
2. Attach the same **ec2-s3-backup** role (Part B step 5).
3. Pull saved progress back down:
   ```bash
   export BUCKET=sudwipto-traffic-backup-2607
   export AWS_DEFAULT_REGION=eu-west-2
   cd ~/group_project
   for d in models logs params results; do aws s3 sync s3://$BUCKET/$d $d; done
   ```
4. Restart the backup loop (Part D) and rerun `MODE=full JOBS=16 ./run_parallel.sh`
   — it skips finished work and continues.

> **Don't forget:** after the run, the S3 bucket keeps costing pennies/month.
> When you've downloaded everything, empty and delete the bucket (S3 → select
> bucket → **Empty**, then **Delete**).

---

## 9. STOP PAYING — the step people forget

Charges continue for as long as the instance exists. When you have your results
downloaded:

- **Terminate** = delete the instance and its disk permanently. Use this when
  you are completely done. **Cheapest — zero further cost.**
  EC2 → select instance → **Instance state → Terminate instance**.

- **Stop** = pause it. Compute billing stops, but you still pay the tiny disk
  cost (~$0.10/day for 30 GB) and can restart later with everything intact. Use
  this if you might run more soon.
  EC2 → select instance → **Instance state → Stop instance**.

> After terminating, double-check the EC2 **Instances** list shows the instance
> as `terminated` and no others are `running`. Also glance at **Elastic IPs**
> and **Volumes** (left menu) — you created none extra here, so they should be
> empty, but a stray unattached volume can quietly bill you.

---

## 10. Set a billing safety net (do this once, before launching)

So a forgotten instance can't run up a surprise bill:

1. Console search → **Billing and Cost Management** → **Budgets** →
   **Create budget**.
2. Choose **Zero spend budget** (emails you at the first cent) or a **Monthly
   cost budget** with a limit like **$25**.
3. Enter your email for alerts → **Create**.

You'll now get an email if spending approaches the limit — your early-warning
that something is still running.

---

## 11. Cost cheat sheet

| Item | Cost |
|------|------|
| c7i.4xlarge On-Demand | ~$0.71 / hour |
| c7i.4xlarge Spot | ~$0.20–0.30 / hour |
| 30 GB gp3 disk | ~$0.10 / day |
| Downloading results (scp) | ~$0.10 total |
| **Full run, On-Demand (~24h)** | **~$17** |
| **Full run, Spot (~24h)** | **~$5–7** |
| **`MODE=overnight` test run (~2–3h)** | **~$2** |
| Instance **terminated** | **$0** |

---

## 12. Troubleshooting quick table

| Symptom | Fix |
|---------|-----|
| `pip install` fails: "No matching distribution found for torch==2.8.0" | System Python too new for the pins. Use the `uv` + Python 3.11 method in §7 — don't use the system `python3`. |
| `sumo` ModuleNotFoundError / `SUMO_HOME` empty | Same cause — deps didn't install because of the Python mismatch. Fix per §7 (uv/3.11), then re-run the `SUMO_HOME=...` line. |
| `uv: command not found` after install | PATH not updated. `export PATH="$HOME/.local/bin:$PATH"` (or `source $HOME/.local/bin/env`). |
| Every job `[FAIL]` in seconds; log shows `libXrender.so.1: cannot open shared object file` or `Could not connect in 1 tries` | Minimal Ubuntu lacks the X11 libs SUMO links. `sudo apt-get install -y libxrender1 libxext6 libxfixes3 libxcursor1 libxrandr2 libxi6 libgl1`, then rerun. |
| Launch fails: "Max spot instance count exceeded" | Spot quota. Check Service Quotas (§5.8); if already ≥16, just retry; else use On-Demand. |
| SSH "Connection timed out" | Security group not allowing your IP. §6 note — add SSH rule for **My IP**. |
| SSH "Permission denied (publickey)" | Wrong username (use `ubuntu`) or wrong/loose key. `chmod 400 ~/.ssh/traffic-key.pem`. |
| "UNPROTECTED PRIVATE KEY FILE" | Run `chmod 400 ~/.ssh/traffic-key.pem`. |
| `c7i.4xlarge` not selectable | Use `c6i.4xlarge` or `m7i.4xlarge`, or try a different Region. |
| Run dies when laptop sleeps | You forgot `tmux`. Always start the run inside `tmux new -s train`. |
| Jobs OOM-killed / machine freezes | Too many parallel jobs for the RAM. Lower it: `JOBS=8 ./run_parallel.sh`. |
| Want to test cheaply first | `MODE=overnight JOBS=16 ./run_parallel.sh` — full flow, small budget. |
| `full` run "stuck" at `[start] tune_*` for many hours; box mostly idle | Not stuck. Tuning is 4 sequential Optuna studies (1 per algo), ~70 min/trial × 30 trials ≈ **35 h**, and only uses 4 cores — the 16-core speedup is the *later* train/eval phase. Confirm progress: `grep -oE '[0-9]+/30 \[[^]]*\]' logs/parallel_*/tune_dqn.log \| tail -1`. Screen may show stale tmux scrollback; check the log file, not the frozen terminal. |
| Lost the `.pem` key | You can't SSH back in. Terminate the instance and start over from §4. |

---

## 13. The whole thing, condensed

```bash
# --- on the AWS website ---
# create account, pick region us-east-1, make key pair "traffic-key",
# launch Ubuntu 24.04 c7i.4xlarge, 30GB disk, allow SSH from My IP.

# --- on your Mac ---
chmod 400 ~/.ssh/traffic-key.pem
ssh -i ~/.ssh/traffic-key.pem ubuntu@PUBLIC_IP

# --- on the cloud machine ---
sudo apt-get update && sudo apt-get install -y git tmux curl \
    libxrender1 libxext6 libxfixes3 libxcursor1 libxrandr2 libxi6 libgl1
git clone https://github.com/sudwiptokm/group-project.git group_project && cd group_project
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
rm -rf venv && uv python install 3.11 && uv venv --python 3.11 venv && source venv/bin/activate
uv pip install -r requirements.txt
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
tmux new -s train
# inside tmux: source venv/bin/activate; export PATH="$HOME/.local/bin:$PATH"; export SUMO_HOME=...
MODE=full JOBS=16 ./run_parallel.sh     # Ctrl-b then d to detach, then close laptop

# --- back on your Mac, after it finishes ---
scp -i ~/.ssh/traffic-key.pem -r ubuntu@PUBLIC_IP:~/group_project/results ./cloud_results
scp -i ~/.ssh/traffic-key.pem -r ubuntu@PUBLIC_IP:~/group_project/logs    ./cloud_logs
scp -i ~/.ssh/traffic-key.pem -r ubuntu@PUBLIC_IP:~/group_project/models  ./cloud_models

# --- on the AWS website ---
# EC2 -> Instance state -> Terminate instance   (STOP PAYING)
```

That's it. Rent → setup → run in tmux → download → terminate.
