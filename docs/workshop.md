## 📖 Workshop Community README

A place where Bittensor miners share, learn, and earn by presenting AI research.

---

### 🎯 Purpose  
Miners are invited to **contribute knowledge** by researching and presenting AI topics or papers (original or survey) in a concise video format. Approved presentations earn a share of Bittensor emissions and drive community learning.

---

### 🗓️ Timeline & Emissions  
- **Submissions open:** Monday, April 28, 2025  
- **Allocation:** 20% of miner emissions reserved for Workshop Community rewards  
- **Earning period:** Starts upon approval and then extends based on community engagement (views & likes)  

---

### 🚀 Submission Process  

1. **Register** as a miner on the subnet (coldkey + hotkey).  
2. **Prepare** a ∼15 min video on an AI research topic or paper. **The video must end with a short introduction to the AI Factory subnet on Bittensor**. This is mandatory to prevent content duplication, such as reuploading videos from others.
3. **Upload** to YouTube with **“AI Factory Bittensor Conference”** phrase in the title. (e.g "Attention Is All You Need" presented by xxx at AI Factory Bittensor Conference)
4. **Submit** on‐chain metadata by running the provided `ai_conference/submit_presentation.py` script with your coldkey, hotkey, and the YouTube video ID.  
5. **Queue & Approval:**  
   - Each coldkey + hotkey pair may queue one video at a time.  
   - Wait for approval before submitting the next.  

---

### 📊 Evaluation & Rewards  

**1. Review Queue**  
- Submissions enter a manual review queue.  
- Subnet team assesses relevance, clarity, and quality.

**2. Approval & Emission**  
- Once approved, your presentation immediately begins to earn emissions.  
- **Score evolves** over time: more **views** & **likes** → extended emission duration.

**3. Multiple Videos**  
- Sequential submissions accumulate scores.  
- Each approved video’s score feeds into the overall Conference Score System.

---

### 🎓 FAQ  

- **Q:** Can I present work that’s not mine?  
  **A:** Yes—survey or overview of others’ research is welcome with proper attribution.

- **Q:** Is 15 min strict?  
  **A:** Aim for ~15 min. Up to 20 min is acceptable, but brevity aids engagement.

- **Q:** Do I need high-end production?  
  **A:** No—clear slides, good audio, and concise delivery are key.

- **Q:** How do I track status?  
  **A:** Watch your miner logs for on-chain events; notify via email if registered.

---

### 💡 Tips for Success  

- **Strong Title & Description:** Use keywords like “Bittensor,” “AI research,” and your topic.  
- **Engaging Thumbnail:** A clear, attractive image drives clicks.  
- **Call to Action:** Ask viewers to like 👍, comment 💬, and subscribe 🔔 to boost emissions.  
- **Share & Promote:** Post in Discord, Twitter, or Medium for early traction.  
- **Use Visuals:** Diagrams, charts, and demos help explain complex ideas.

---

## 📐 Mathematical Scoring Model  
```markdown
Let v = total views, l = total likes.

# Time-window parameter (tau)
# tau ∈ [3, 7]
tau = 3 + 4 * min((0.18 * v + l) / 10000, 1)

# Daily decay margin for day i in [MIN_DAY, MAX_DAY]
decay(i) = tau - i

# Tick increment calculation
# S = scoring interval in seconds (configured on-chain)
# One day = 86400 seconds
ticks_total = (MAX_DAY * 86400) / S
# Each valid tick adds:
Delta_i = 1 / ticks_total  if decay(i) >= 0
         0                 otherwise

# Scores Delta_i accumulate over all valid days
to form the Conference Score
```
