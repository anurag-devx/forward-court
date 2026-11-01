# Forward Court demo set

Three stable samples, one per input type. Media files stay out of git
(see manifest.json files for sources); rerun with the commands below.

## 1. Text — lemon-cancer forward (FALSE via ClaimReview)

Input (`tests/samples/english.txt`):

> Japanese doctors have confirmed that drinking hot water with lemon every
> morning destroys cancer cells completely. No need for chemo anymore!
> Please share with everyone to save lives.

Command:

```
.venv/bin/python main.py "$(cat tests/samples/english.txt)"
```

- Verdict: FALSE, confidence high, via ClaimReview short-circuit.
- Reason: Already fact-checked by vishvasnews.com: rated False.
- Key source: https://www.vishvasnews.com/english/health/fact-check-false-claims-of-curing-cancer-getting-viral/
- Time: ~67s, ~70k tokens.

## 2. Image — Pithoragarh bridge collapse (MISLEADING, stable 2/2)

Sample `57-0-www.vishvasnews.com.jpg` (expected Misleading).

Caption:

> उत्तराखंड के पिथौरागढ़ में ओवरलोड ट्रक गुजरते ही भरभराकर ढह गया पुल।

Command:

```
.venv/bin/python main.py --image tests/samples/images/57-0-www.vishvasnews.com.jpg \
  --caption "उत्तराखंड के पिथौरागढ़ में ओवरलोड ट्रक गुजरते ही भरभराकर ढह गया पुल।"
```

- Verdict: MISLEADING, confidence high, via ClaimReview (Fact Crescendo),
  stable across 2 runs.
- Reason: वायरल वीडियो में पिथौरागढ़ में ओवरलोड ट्रक के कारण पुल ढहने की
  घटना वास्तव में पुरानी है, हाल की नहीं। फैक्ट-चेक के अनुसार यह 2020 का
  पुराना वीडियो है जिसे हाल की घटना बताकर वायरल किया जा रहा है, इसलिए इसे
  नया बताकर पेश करना भ्रामक है।
- Key source: https://english.factcrescendo.com/2026/10/27/old-video-of-a-bridge-collapse-in-pithoragarh-in-uttarakhand-goes-viral-as-recent/
- Time: ~181s / ~219s over 2 runs.

## 3. Video — Cyclone Arnab old clip (MISLEADING, stable verdict 2/2)

Sample `demo04-newschecker.in.mp4`, original upload 2026-07-06
(https://www.youtube.com/watch?v=6YpYoP71-F4).

Caption (full viral claim):

> A video showing trees swaying amid powerful winds is being shared on
> social media as footage of Cyclone Arab wreaking havoc in Visakhapatnam,
> Andhra Pradesh.

Command:

```
.venv/bin/python main.py --video tests/samples/demo/demo04-newschecker.in.mp4 \
  --caption "A video showing trees swaying amid powerful winds is being shared on social media as footage of Cyclone Arab wreaking havoc in Visakhapatnam, Andhra Pradesh."
```

- Verdict: MISLEADING, confidence high, via debate + 3/3 judge vote, stable
  across 2 runs (recycled override fired in run 1).
- Reason: The clip does show trees swaying amid powerful winds, but
  fact-checks identify it as an old video recycled as footage of Cyclone
  Arab/Arnab wreaking havoc in Visakhapatnam, Andhra Pradesh. The same
  tree-swaying clip was previously falsely linked to Cyclone Biparjoy.
- Key source: https://english.factcrescendo.com/2026/10/28/old-video-goes-viral-linking-to-cyclone-arnab-in-visakhapatnam-ap/
- Time: ~155s / ~143s over 2 runs.
