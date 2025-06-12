Before running the test code, you need to first download the datasets from the official websites of each benchmark. The data directory structure should be formatted as follows:

```
DATASET_DIR/
├── LongVideoBench
│   ├── videos
│   ├── subtitles
│   ├── qa_val.json
...
```

The `qa_val.json` file contains a list where each element is a dictionary with the following format:

```json
{
    "video_id": "soNQYQXrx_A",
    "question": "A man dressed in a black suit and white shirt with a tie is floating on the surface of the water. What happens to his clothes when the subtitle says 'driven out of the world'?",
    "question_wo_referring_query": "What happens to this man's clothes?",
    "correct_choice": 4,
    "position": [
        4030,
        7571
    ],
    "topic_category": "Recreational: MR-Movie-Recaps",
    "question_category": "TAA",
    "level": "L2-Relation",
    "id": "soNQYQXrx_A_1",
    "video_path": "~/benchmark/formulated/LongVideoBench/videos/soNQYQXrx_A.mp4",
    "subtitle_path": "soNQYQXrx_A_en.json",
    "duration_group": 600,
    "starting_timestamp_for_subtitles": 0,
    "duration": 563.37,
    "view_count": 93803,
    "options": [
        "A.He puts on a denim jacket.",
        "B.He puts on a white tank top.",
        "C.He puts on a black bathrobe.",
        "D.He puts on a green short-sleeve shirt.",
        "E.He is shirtless."
    ],
    "answer": "E",
    "task": "TAA"
}
```

After preparing the data, you can run the corresponding preprocessing code and test code according to the instructions in `eval.sh`.