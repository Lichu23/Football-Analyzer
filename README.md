# Football Analyzer

Football coaching staffs spend hours manually reviewing match footage to understand player positioning, movement patterns, and team shape. This process is slow, subjective, and inaccessible for clubs without dedicated analysts.

**Football Analyzer automates that work** — feed it a match video and it produces objective, visual data about every player on the pitch.

---

## The problem

- Tactical analysis requires watching full match footage frame by frame
- Small and mid-sized clubs can't afford dedicated analysts or expensive software (Hudl, Wyscout)
- Coaches make positioning decisions based on memory and intuition rather than data

## The solution

An ML pipeline that processes raw video and outputs position-based analytics per player — starting with heatmaps and expanding into speed, sprints, team shape, and ball tracking.

---

## How it works

1. **Detection** — YOLOv8 identifies every person on the pitch in each frame
2. **Tracking** — ByteTrack assigns a consistent ID to each player across frames
3. **Mapping** — player foot positions are normalized and mapped onto a standard 105×68m pitch diagram
4. **Output** — one heatmap per player showing where they spent the most time during the match

---

## Technologies

| Layer | Technology |
|---|---|
| Object detection | [YOLOv8](https://github.com/ultralytics/ultralytics) |
| Multi-object tracking | [ByteTrack](https://github.com/roboflow/supervision) via Supervision |
| Video processing | OpenCV |
| Visualization | Matplotlib, NumPy, SciPy |
| Runtime | Python 3.10+ |

---

## Roadmap

- [x] Player heatmaps
- [ ] Distance covered, speed, sprint detection
- [ ] Team classification by jersey color
- [ ] Ball tracking, pass networks, Voronoi diagrams
- [ ] Multi-video support (first half + second half)
- [ ] SaaS version — web upload, background processing, results dashboard
