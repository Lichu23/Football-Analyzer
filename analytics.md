# Football Analyzer — Roadmap

---

## LOCAL VERSION (validate the ML core first)

### Phase 1 — Player heatmap (current goal)
- [x] **Player heatmap** — shows where each player spent the most time across the video, overlaid on a pitch diagram. One image per player, labeled by tracking ID.

### Phase 2 — Movement metrics
- [ ] **Distance covered** — total meters each player ran during the video
- [ ] **Speed** — average and max speed per player (km/h), calculated from position changes between frames
- [ ] **Sprint detection** — flags moments where a player exceeds a speed threshold (e.g. >20 km/h)
- [ ] **Player trajectory** — draws the path each player took across the pitch

### Phase 3 — Team analytics
- [ ] **Team separation** — classify players into two teams using jersey color (KMeans clustering)
- [ ] **Team heatmap** — combined heatmap per team instead of per individual player
- [ ] **Possession zones** — which team controlled which areas of the pitch most
- [ ] **Defensive/offensive line** — average horizontal position of each team's line per moment

### Phase 4 — Advanced analytics
- [ ] **Ball tracking** — detect and track the ball separately from players
- [ ] **Pass network** — which players were close to each other and when (proximity graph)
- [ ] **Voronoi diagram** — shows each player's area of coverage/dominance at a given moment
- [ ] **Zone time breakdown** — how much time each player spent in each third (defensive, middle, attacking)
- [ ] **Jersey number detection** — map tracking IDs to actual jersey numbers using OCR

### Phase 5 — Multi-video support
- [ ] **Two video input** — accept first half + second half as separate files
- [ ] **Combined heatmap** — merge heatmaps from both halves for the same player
- [ ] **Half comparison** — side-by-side heatmap first half vs second half per player

---

## SAAS VERSION (after local logic is validated)

### Phase 6 — Backend API
- [ ] Wrap ML code in **FastAPI** — expose endpoints to upload video and retrieve results
- [ ] **Job queue** with Celery + Redis — video processing runs in background, user is not blocked
- [ ] **File storage** with AWS S3 — store uploaded videos and output files (heatmaps, annotated video)
- [ ] Processing status endpoint — frontend can poll "is my job done yet?"
- [ ] Auth system — user accounts, API keys

### Phase 7 — Frontend (web UI)
- [ ] **Upload page** — drag and drop video file, select analysis options
- [ ] **Processing screen** — progress bar or status indicator while job runs
- [ ] **Results page** — displays heatmap images per player, annotated video, downloadable exports
- [ ] **Dashboard** — history of past analyzed videos per user

### Phase 8 — Monetization
- [ ] **Stripe integration** — payments for plans or pay-per-video
- [ ] **Freemium gate** — 1 free video, paid plan for unlimited
- [ ] **Subscription plans** — individual coach vs club/academy tier
- [ ] **Usage limits** — cap free tier by video length or number of players

### Phase 9 — Scale & reliability
- [ ] **GPU worker** — move inference to a GPU server (AWS EC2 g4dn or RunPod) for faster processing
- [ ] **Queue monitoring** — track job failures, retries, processing time
- [ ] **Multi-tenant isolation** — each user's data is private and separate
- [ ] **CDN** for output files — fast delivery of heatmap images and videos globally
