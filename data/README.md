# Data Directory

This directory holds data assets for Curio. Large files (images, datasets) are **not** checked into git.

## Structure

```
data/
├── README.md           # This file
├── articles/           # Article text files (Wikipedia intros)
│   ├── dev/            # 100 articles + 15 no-match for development
│   └── test/           # 100 articles + 15 no-match for final evaluation
└── images/             # Image files (downloaded via scripts, gitignored)
    ├── corpus/         # 3,000-image Unsplash Lite subset
    └── holdback/       # 200 images for freshness testing
```

## Data Sources

- **Images:** 3,000-image subset of the [Unsplash Lite dataset](https://github.com/unsplash/datasets) across ~20 categories, including confusable groups (fox/wolf/coyote, leopard/cheetah/jaguar, crow/raven, alligator/crocodile).
- **Articles:** 200 real article intros from Wikipedia (CC BY-SA), plus 30 "no-match" articles on topics the image corpus deliberately excludes.
- **Hard negatives:** 40 article–image pairs where the image is a confusable lookalike.

## Download

```bash
# Download images and articles
uv run python -m curio.scripts.download_data
```
