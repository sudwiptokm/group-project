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
   option** → check **Spot**. Skip this for your first run.
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

```bash
# 7.1 system packages
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git tmux

# 7.2 get the code (public repo clone; for a private repo see note below)
git clone https://github.com/sudwiptokm/group-project.git group_project
cd group_project

# 7.3 python environment + dependencies (a few minutes)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 7.4 tell the tools where SUMO lives
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
```

> **Private repo?** Easiest is to make the GitHub repo public temporarily, or
> create a GitHub **Personal Access Token** and clone with
> `git clone https://<TOKEN>@github.com/sudwiptokm/group-project.git`.
> Do NOT put your token in any committed file.

### Start the run inside `tmux` (survives disconnects)
`tmux` keeps the job alive even if your laptop sleeps or SSH drops. Without it, a
dropped connection kills a multi-hour run.

```bash
tmux new -s train        # opens a persistent session

# inside tmux:
source venv/bin/activate
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
MODE=full JOBS=16 ./run_parallel.sh
```

- `MODE=full` = the publication-size budget. Use `MODE=overnight` for a quick
  smaller run first if you want to test the whole flow cheaply.
- `JOBS=16` = run 16 jobs at once (matches the 16 vCPUs).

**Detach** (leave it running, return to normal prompt): press `Ctrl-b`, release,
then press `d`. You can now safely close your laptop or disconnect.

**Reconnect later:** SSH back in (§6), then:
```bash
tmux attach -t train
```
You'll see live progress. Detach again with `Ctrl-b` then `d`.

### Watching progress
The driver prints one line per job (`[start]`, `[ok]`, `[FAIL]`). Detailed
per-job logs are in `logs/parallel_<timestamp>/`. To tail the driver log from a
second SSH window:
```bash
tail -f logs/parallel_*/driver.log
```

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

### Spot interruption (only if you chose Spot in §5.8)
If AWS reclaims a Spot instance mid-run, the instance stops. The script is
**resumable**: launch a new instance, redo §7 setup, and run the exact same
`MODE=full JOBS=16 ./run_parallel.sh` — it skips every result that already
exists and continues. (For this to help, the disk must persist or you re-clone;
simplest is just to let it redo missing pieces.)

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
| SSH "Connection timed out" | Security group not allowing your IP. §6 note — add SSH rule for **My IP**. |
| SSH "Permission denied (publickey)" | Wrong username (use `ubuntu`) or wrong/loose key. `chmod 400 ~/.ssh/traffic-key.pem`. |
| "UNPROTECTED PRIVATE KEY FILE" | Run `chmod 400 ~/.ssh/traffic-key.pem`. |
| `c7i.4xlarge` not selectable | Use `c6i.4xlarge` or `m7i.4xlarge`, or try a different Region. |
| Run dies when laptop sleeps | You forgot `tmux`. Always start the run inside `tmux new -s train`. |
| Jobs OOM-killed / machine freezes | Too many parallel jobs for the RAM. Lower it: `JOBS=8 ./run_parallel.sh`. |
| Want to test cheaply first | `MODE=overnight JOBS=16 ./run_parallel.sh` — full flow, small budget. |
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
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip git tmux
git clone https://github.com/sudwiptokm/group-project.git group_project && cd group_project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export SUMO_HOME=$(python -c 'import sumo; print(sumo.SUMO_HOME)')
tmux new -s train
MODE=full JOBS=16 ./run_parallel.sh     # Ctrl-b then d to detach

# --- back on your Mac, after it finishes ---
scp -i ~/.ssh/traffic-key.pem -r ubuntu@PUBLIC_IP:~/group_project/results ./cloud_results
scp -i ~/.ssh/traffic-key.pem -r ubuntu@PUBLIC_IP:~/group_project/logs    ./cloud_logs
scp -i ~/.ssh/traffic-key.pem -r ubuntu@PUBLIC_IP:~/group_project/models  ./cloud_models

# --- on the AWS website ---
# EC2 -> Instance state -> Terminate instance   (STOP PAYING)
```

That's it. Rent → setup → run in tmux → download → terminate.
