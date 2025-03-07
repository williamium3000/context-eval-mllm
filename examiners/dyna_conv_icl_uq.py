from PIL import Image

from utils.vg import load_vg, format_case_vg
from utils.llm import LLMChat, parse_json
from examiners import prompt as PROMPT
from infer.infer_llava import load_model, eval_model
import os
import argparse
import json
import tqdm
import copy

SYSTEM_PROMPT = \
"""
Task: Your task is to have multiple conversations with a vision-language model. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. You need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should test model's ability to answer unanswerable questions. An unanswerable question refers to the question that the model cannot answer based on the given input image because the queried information is missing, unclear, or speculative.
3. You ask mostly asks the model questions that are unanswerable following the roughly 4 categories: 
(1) Nonexistent objects, attributes and relationships;
(2) background details about objects not depicted in the image;
(3) questions about events or conditions that occurred before or after the moment captured in the image;
(4) missing visual information that are visually unclear, hidden, or blurred in the image.
4. DO NOT ask existence questions such as "Are there any ...?". Directly ask questions of the nonexistent information.
5. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
6. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 


You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_PROMPT = \
"""
Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

icl_cases = [
    {
        "image_id": 3,
        "url": "https://cs.stanford.edu/people/rak248/VG_100K/3.jpg",
        "width": 640,
        "height": 480,
        "objects": [
            {
                "object_id": 5091,
                "x": 456,
                "y": 266,
                "w": 98,
                "h": 79,
                "names": [
                    "office phone"
                ],
                "synsets": []
            },
            {
                "object_id": 5092,
                "x": 83,
                "y": 202,
                "w": 17,
                "h": 24,
                "names": [
                    "outlet"
                ],
                "synsets": [
                    "mercantile_establishment.n.01"
                ]
            },
            {
                "object_id": 5093,
                "x": 42,
                "y": 207,
                "w": 22,
                "h": 29,
                "names": [
                    "outlet"
                ],
                "synsets": [
                    "mercantile_establishment.n.01"
                ]
            },
            {
                "object_id": 1060286,
                "x": 168,
                "y": 273,
                "w": 175,
                "h": 48,
                "names": [
                    "keyboard"
                ],
                "synsets": [
                    "keyboard.n.01"
                ]
            },
            {
                "object_id": 1060246,
                "x": 159,
                "y": 119,
                "w": 149,
                "h": 155,
                "names": [
                    "monitor",
                    "computer"
                ],
                "synsets": [
                    "monitor.n.04"
                ]
            },
            {
                "object_id": 5096,
                "x": 212,
                "y": 340,
                "w": 88,
                "h": 139,
                "names": [
                    "cpu"
                ],
                "synsets": [
                    "central_processing_unit.n.01"
                ]
            },
            {
                "object_id": 5097,
                "x": 129,
                "y": 233,
                "w": 510,
                "h": 244,
                "names": [
                    "desktop"
                ],
                "synsets": [
                    "desktop.n.01"
                ]
            },
            {
                "object_id": 5098,
                "x": 2,
                "y": 269,
                "w": 139,
                "h": 201,
                "names": [
                    "filing cabinet"
                ],
                "synsets": [
                    "cabinet.n.01"
                ]
            },
            {
                "object_id": 5099,
                "x": 48,
                "y": 423,
                "w": 94,
                "h": 42,
                "names": [
                    "drawer"
                ],
                "synsets": [
                    "drawer.n.01"
                ]
            },
            {
                "object_id": 1060251,
                "x": 2,
                "y": 321,
                "w": 128,
                "h": 123,
                "names": [
                    "drawer"
                ],
                "synsets": [
                    "drawer.n.01"
                ]
            },
            {
                "object_id": 5101,
                "x": 574,
                "y": 270,
                "w": 65,
                "h": 134,
                "names": [
                    "computer case"
                ],
                "synsets": []
            },
            {
                "object_id": 1060253,
                "x": 346,
                "y": 296,
                "w": 29,
                "h": 28,
                "names": [
                    "mouse"
                ],
                "synsets": [
                    "mouse.n.01"
                ]
            },
            {
                "object_id": 5103,
                "x": 465,
                "y": 268,
                "w": 38,
                "h": 61,
                "names": [
                    "wireless phone"
                ],
                "synsets": [
                    "telephone.n.01"
                ]
            },
            {
                "object_id": 5104,
                "x": 465,
                "y": 277,
                "w": 73,
                "h": 66,
                "names": [
                    "office phone base"
                ],
                "synsets": []
            },
            {
                "object_id": 1060282,
                "x": 130,
                "y": 235,
                "w": 506,
                "h": 241,
                "names": [
                    "desk"
                ],
                "synsets": [
                    "desk.n.01"
                ]
            },
            {
                "object_id": 5106,
                "x": 42,
                "y": 190,
                "w": 89,
                "h": 48,
                "names": [
                    "multiple outlet"
                ],
                "synsets": [
                    "mercantile_establishment.n.01"
                ]
            },
            {
                "object_id": 5107,
                "x": 43,
                "y": 208,
                "w": 26,
                "h": 33,
                "names": [
                    "plug"
                ],
                "synsets": [
                    "plug.n.01"
                ]
            },
            {
                "object_id": 5108,
                "x": 66,
                "y": 207,
                "w": 15,
                "h": 21,
                "names": [
                    "plug"
                ],
                "synsets": [
                    "plug.n.01"
                ]
            },
            {
                "object_id": 5109,
                "x": 85,
                "y": 200,
                "w": 16,
                "h": 25,
                "names": [
                    "plug"
                ],
                "synsets": [
                    "plug.n.01"
                ]
            },
            {
                "object_id": 5110,
                "x": 106,
                "y": 200,
                "w": 17,
                "h": 23,
                "names": [
                    "plug"
                ],
                "synsets": [
                    "plug.n.01"
                ]
            },
            {
                "object_id": 1060269,
                "x": 544,
                "y": 89,
                "w": 94,
                "h": 149,
                "names": [
                    "girl",
                    "woman"
                ],
                "synsets": [
                    "girl.n.01"
                ]
            },
            {
                "object_id": 1060270,
                "x": 417,
                "y": 63,
                "w": 82,
                "h": 50,
                "names": [
                    "monitor",
                    "calendar"
                ],
                "synsets": [
                    "monitor.n.04"
                ]
            },
            {
                "object_id": 1060274,
                "x": 3,
                "y": 6,
                "w": 258,
                "h": 265,
                "names": [
                    "wall"
                ],
                "synsets": [
                    "wall.n.01"
                ]
            },
            {
                "object_id": 5114,
                "x": 585,
                "y": 98,
                "w": 55,
                "h": 109,
                "names": [
                    "hair"
                ],
                "synsets": [
                    "hair.n.01"
                ]
            },
            {
                "object_id": 5115,
                "x": 593,
                "y": 166,
                "w": 34,
                "h": 17,
                "names": [
                    "chain"
                ],
                "synsets": [
                    "chain.n.01"
                ]
            },
            {
                "object_id": 1060281,
                "x": 405,
                "y": 323,
                "w": 53,
                "h": 16,
                "names": [
                    "pen"
                ],
                "synsets": [
                    "pen.n.01"
                ]
            },
            {
                "object_id": 5117,
                "x": 137,
                "y": 408,
                "w": 52,
                "h": 58,
                "names": [
                    "cable"
                ],
                "synsets": [
                    "cable.n.01"
                ]
            },
            {
                "object_id": 5118,
                "x": 110,
                "y": 340,
                "w": 390,
                "h": 139,
                "names": [
                    "floor"
                ],
                "synsets": [
                    "floor.n.01"
                ]
            },
            {
                "object_id": 5119,
                "x": 150,
                "y": 387,
                "w": 84,
                "h": 40,
                "names": [
                    "cable"
                ],
                "synsets": [
                    "cable.n.01"
                ]
            },
            {
                "object_id": 1060261,
                "x": 42,
                "y": 190,
                "w": 89,
                "h": 48,
                "names": [
                    "pluged",
                    "plugged",
                    "outlets"
                ],
                "synsets": [
                    "mercantile_establishment.n.01"
                ]
            },
            {
                "object_id": 1060252,
                "x": 574,
                "y": 270,
                "w": 65,
                "h": 134,
                "names": [
                    "bag"
                ],
                "synsets": [
                    "bag.n.01"
                ]
            },
            {
                "object_id": 1060287,
                "x": 304,
                "y": 30,
                "w": 217,
                "h": 97,
                "names": [
                    "wall"
                ],
                "synsets": [
                    "wall.n.01"
                ]
            },
            {
                "object_id": 1060277,
                "x": 585,
                "y": 98,
                "w": 55,
                "h": 109,
                "names": [
                    "black hair",
                    "dark hair"
                ],
                "synsets": [
                    "hair.n.01"
                ]
            },
            {
                "object_id": 1060242,
                "x": 456,
                "y": 266,
                "w": 98,
                "h": 79,
                "names": [
                    "telephone",
                    "phone"
                ],
                "synsets": [
                    "telephone.n.01"
                ]
            },
            {
                "object_id": 1060288,
                "x": 538,
                "y": 174,
                "w": 101,
                "h": 68,
                "names": [
                    "white top",
                    "white t-shirt"
                ],
                "synsets": []
            },
            {
                "object_id": 1060289,
                "x": 570,
                "y": 333,
                "w": 69,
                "h": 81,
                "names": [
                    "strap"
                ],
                "synsets": [
                    "strap.n.01"
                ]
            },
            {
                "object_id": 1060291,
                "x": 125,
                "y": 71,
                "w": 513,
                "h": 402,
                "names": [
                    "cubicles"
                ],
                "synsets": [
                    "booth.n.02"
                ]
            },
            {
                "object_id": 1060290,
                "x": 243,
                "y": 81,
                "w": 300,
                "h": 204,
                "names": [
                    "dividing screen",
                    "partition"
                ],
                "synsets": [
                    "partition.n.01"
                ]
            },
            {
                "object_id": 1060292,
                "x": 210,
                "y": 2,
                "w": 61,
                "h": 44,
                "names": [
                    "picture"
                ],
                "synsets": [
                    "picture.n.01"
                ]
            },
            {
                "object_id": 1060294,
                "x": 421,
                "y": 105,
                "w": 99,
                "h": 95,
                "names": [
                    "picture",
                    "calendar"
                ],
                "synsets": [
                    "calendar.n.01"
                ]
            },
            {
                "object_id": 1060293,
                "x": 413,
                "y": 56,
                "w": 101,
                "h": 57,
                "names": [
                    "computer"
                ],
                "synsets": [
                    "computer.n.01"
                ]
            },
            {
                "object_id": 1060295,
                "x": 59,
                "y": 211,
                "w": 81,
                "h": 89,
                "names": [
                    "cables"
                ],
                "synsets": [
                    "cable.n.01"
                ]
            },
            {
                "object_id": 1060297,
                "x": 3,
                "y": 2,
                "w": 552,
                "h": 65,
                "names": [
                    "wall"
                ],
                "synsets": [
                    "wall.n.01"
                ]
            },
            {
                "object_id": 1060296,
                "x": 209,
                "y": 3,
                "w": 341,
                "h": 45,
                "names": [
                    "photos"
                ],
                "synsets": [
                    "photograph.n.01"
                ]
            },
            {
                "object_id": 1060248,
                "x": 129,
                "y": 233,
                "w": 510,
                "h": 244,
                "names": [
                    "table"
                ],
                "synsets": [
                    "table.n.02"
                ]
            },
            {
                "object_id": 1060298,
                "x": 18,
                "y": 353,
                "w": 81,
                "h": 44,
                "names": [
                    "handle"
                ],
                "synsets": [
                    "handle.n.01"
                ]
            },
            {
                "object_id": 1060299,
                "x": 125,
                "y": 376,
                "w": 141,
                "h": 95,
                "names": [
                    "floor"
                ],
                "synsets": [
                    "floor.n.01"
                ]
            },
            {
                "object_id": 1060300,
                "x": 585,
                "y": 169,
                "w": 32,
                "h": 15,
                "names": [
                    "necklace"
                ],
                "synsets": [
                    "necklace.n.01"
                ]
            },
            {
                "object_id": 1060245,
                "x": 168,
                "y": 273,
                "w": 175,
                "h": 48,
                "names": [
                    "grey keys"
                ],
                "synsets": [
                    "key.n.01"
                ]
            },
            {
                "object_id": 1060301,
                "x": 532,
                "y": 156,
                "w": 52,
                "h": 52,
                "names": [
                    "desk"
                ],
                "synsets": [
                    "desk.n.01"
                ]
            },
            {
                "object_id": 1060247,
                "x": 212,
                "y": 340,
                "w": 88,
                "h": 139,
                "names": [
                    "computer tower"
                ],
                "synsets": []
            },
            {
                "object_id": 1060249,
                "x": 2,
                "y": 269,
                "w": 139,
                "h": 201,
                "names": [
                    "cabinet"
                ],
                "synsets": [
                    "cabinet.n.01"
                ]
            }
        ],
        "attributes": [
            {
                "object_id": 5091,
                "x": 456,
                "y": 266,
                "w": 98,
                "h": 79,
                "names": [
                    "office phone"
                ],
                "synsets": [],
                "attributes": [
                    "multi-line phone"
                ]
            },
            {
                "object_id": 5092,
                "x": 83,
                "y": 202,
                "w": 17,
                "h": 24,
                "names": [
                    "outlet"
                ],
                "synsets": [
                    "mercantile_establishment.n.01"
                ],
                "attributes": [
                    "electrical"
                ]
            },
            {
                "object_id": 5093,
                "x": 42,
                "y": 207,
                "w": 22,
                "h": 29,
                "names": [
                    "outlet"
                ],
                "synsets": [
                    "mercantile_establishment.n.01"
                ],
                "attributes": [
                    "data line"
                ]
            },
            {
                "object_id": 1060286,
                "x": 168,
                "y": 273,
                "w": 175,
                "h": 48,
                "names": [
                    "keyboard"
                ],
                "synsets": [
                    "keyboard.n.01"
                ],
                "attributes": [
                    "white"
                ]
            },
            {
                "object_id": 1060246,
                "x": 159,
                "y": 119,
                "w": 149,
                "h": 155,
                "names": [
                    "monitor",
                    "computer"
                ],
                "synsets": [
                    "monitor.n.04"
                ],
                "attributes": [
                    "white",
                    "switched off",
                    "turned off"
                ]
            },
            {
                "object_id": 5096,
                "x": 212,
                "y": 340,
                "w": 88,
                "h": 139,
                "names": [
                    "cpu"
                ],
                "synsets": [
                    "central_processing_unit.n.01"
                ],
                "attributes": [
                    "computer tower"
                ]
            },
            {
                "object_id": 5097,
                "x": 129,
                "y": 233,
                "w": 510,
                "h": 244,
                "names": [
                    "desktop"
                ],
                "synsets": [
                    "desktop.n.01"
                ],
                "attributes": [
                    "curved"
                ]
            },
            {
                "object_id": 5098,
                "x": 2,
                "y": 269,
                "w": 139,
                "h": 201,
                "names": [
                    "filing cabinet"
                ],
                "synsets": [
                    "cabinet.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5099,
                "x": 48,
                "y": 423,
                "w": 94,
                "h": 42,
                "names": [
                    "drawer"
                ],
                "synsets": [
                    "drawer.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060251,
                "x": 2,
                "y": 321,
                "w": 128,
                "h": 123,
                "names": [
                    "drawer"
                ],
                "synsets": [
                    "drawer.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5101,
                "x": 574,
                "y": 270,
                "w": 65,
                "h": 134,
                "names": [
                    "computer case"
                ],
                "synsets": [],
                "attributes": [
                    "black",
                    "leather"
                ]
            },
            {
                "object_id": 1060253,
                "x": 346,
                "y": 296,
                "w": 29,
                "h": 28,
                "names": [
                    "mouse"
                ],
                "synsets": [
                    "mouse.n.01"
                ],
                "attributes": [
                    "white"
                ]
            },
            {
                "object_id": 5103,
                "x": 465,
                "y": 268,
                "w": 38,
                "h": 61,
                "names": [
                    "wireless phone"
                ],
                "synsets": [
                    "telephone.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5104,
                "x": 465,
                "y": 277,
                "w": 73,
                "h": 66,
                "names": [
                    "office phone base"
                ],
                "synsets": [],
                "attributes": None
            },
            {
                "object_id": 1060282,
                "x": 130,
                "y": 235,
                "w": 506,
                "h": 241,
                "names": [
                    "desk"
                ],
                "synsets": [
                    "desk.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5106,
                "x": 42,
                "y": 190,
                "w": 89,
                "h": 48,
                "names": [
                    "multiple outlet"
                ],
                "synsets": [
                    "mercantile_establishment.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5107,
                "x": 43,
                "y": 208,
                "w": 26,
                "h": 33,
                "names": [
                    "plug"
                ],
                "synsets": [
                    "plug.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5108,
                "x": 66,
                "y": 207,
                "w": 15,
                "h": 21,
                "names": [
                    "plug"
                ],
                "synsets": [
                    "plug.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5109,
                "x": 85,
                "y": 200,
                "w": 16,
                "h": 25,
                "names": [
                    "plug"
                ],
                "synsets": [
                    "plug.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5110,
                "x": 106,
                "y": 200,
                "w": 17,
                "h": 23,
                "names": [
                    "plug"
                ],
                "synsets": [
                    "plug.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060269,
                "x": 544,
                "y": 89,
                "w": 94,
                "h": 149,
                "names": [
                    "girl",
                    "woman"
                ],
                "synsets": [
                    "girl.n.01"
                ],
                "attributes": [
                    "sitting"
                ]
            },
            {
                "object_id": 1060270,
                "x": 417,
                "y": 63,
                "w": 82,
                "h": 50,
                "names": [
                    "monitor",
                    "calendar"
                ],
                "synsets": [
                    "monitor.n.04"
                ],
                "attributes": [
                    "off"
                ]
            },
            {
                "object_id": 1060274,
                "x": 3,
                "y": 6,
                "w": 258,
                "h": 265,
                "names": [
                    "wall"
                ],
                "synsets": [
                    "wall.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5114,
                "x": 585,
                "y": 98,
                "w": 55,
                "h": 109,
                "names": [
                    "hair"
                ],
                "synsets": [
                    "hair.n.01"
                ],
                "attributes": [
                    "long"
                ]
            },
            {
                "object_id": 5115,
                "x": 593,
                "y": 166,
                "w": 34,
                "h": 17,
                "names": [
                    "chain"
                ],
                "synsets": [
                    "chain.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060281,
                "x": 405,
                "y": 323,
                "w": 53,
                "h": 16,
                "names": [
                    "pen"
                ],
                "synsets": [
                    "pen.n.01"
                ],
                "attributes": [
                    "blue",
                    "yellow",
                    "black"
                ]
            },
            {
                "object_id": 5117,
                "x": 137,
                "y": 408,
                "w": 52,
                "h": 58,
                "names": [
                    "cable"
                ],
                "synsets": [
                    "cable.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5118,
                "x": 110,
                "y": 340,
                "w": 390,
                "h": 139,
                "names": [
                    "floor"
                ],
                "synsets": [
                    "floor.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 5119,
                "x": 150,
                "y": 387,
                "w": 84,
                "h": 40,
                "names": [
                    "cable"
                ],
                "synsets": [
                    "cable.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060261,
                "x": 42,
                "y": 190,
                "w": 89,
                "h": 48,
                "names": [
                    "pluged",
                    "plugged",
                    "outlets"
                ],
                "synsets": [
                    "mercantile_establishment.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060252,
                "x": 574,
                "y": 270,
                "w": 65,
                "h": 134,
                "names": [
                    "bag"
                ],
                "synsets": [
                    "bag.n.01"
                ],
                "attributes": [
                    "black"
                ]
            },
            {
                "object_id": 1060287,
                "x": 304,
                "y": 30,
                "w": 217,
                "h": 97,
                "names": [
                    "wall"
                ],
                "synsets": [
                    "wall.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060277,
                "x": 585,
                "y": 98,
                "w": 55,
                "h": 109,
                "names": [
                    "black hair",
                    "dark hair"
                ],
                "synsets": [
                    "hair.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060242,
                "x": 456,
                "y": 266,
                "w": 98,
                "h": 79,
                "names": [
                    "telephone",
                    "phone"
                ],
                "synsets": [
                    "telephone.n.01"
                ],
                "attributes": [
                    "black"
                ]
            },
            {
                "object_id": 1060288,
                "x": 538,
                "y": 174,
                "w": 101,
                "h": 68,
                "names": [
                    "white top",
                    "white t-shirt"
                ],
                "synsets": [],
                "attributes": None
            },
            {
                "object_id": 1060289,
                "x": 570,
                "y": 333,
                "w": 69,
                "h": 81,
                "names": [
                    "strap"
                ],
                "synsets": [
                    "strap.n.01"
                ],
                "attributes": [
                    "black"
                ]
            },
            {
                "object_id": 1060291,
                "x": 125,
                "y": 71,
                "w": 513,
                "h": 402,
                "names": [
                    "cubicles"
                ],
                "synsets": [
                    "booth.n.02"
                ],
                "attributes": None
            },
            {
                "object_id": 1060290,
                "x": 243,
                "y": 81,
                "w": 300,
                "h": 204,
                "names": [
                    "dividing screen",
                    "partition"
                ],
                "synsets": [
                    "partition.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060292,
                "x": 210,
                "y": 2,
                "w": 61,
                "h": 44,
                "names": [
                    "picture"
                ],
                "synsets": [
                    "picture.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060294,
                "x": 421,
                "y": 105,
                "w": 99,
                "h": 95,
                "names": [
                    "picture",
                    "calendar"
                ],
                "synsets": [
                    "calendar.n.01"
                ],
                "attributes": [
                    "hanged"
                ]
            },
            {
                "object_id": 1060293,
                "x": 413,
                "y": 56,
                "w": 101,
                "h": 57,
                "names": [
                    "computer"
                ],
                "synsets": [
                    "computer.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060295,
                "x": 59,
                "y": 211,
                "w": 81,
                "h": 89,
                "names": [
                    "cables"
                ],
                "synsets": [
                    "cable.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060297,
                "x": 3,
                "y": 2,
                "w": 552,
                "h": 65,
                "names": [
                    "wall"
                ],
                "synsets": [
                    "wall.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060296,
                "x": 209,
                "y": 3,
                "w": 341,
                "h": 45,
                "names": [
                    "photos"
                ],
                "synsets": [
                    "photograph.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060248,
                "x": 129,
                "y": 233,
                "w": 510,
                "h": 244,
                "names": [
                    "table"
                ],
                "synsets": [
                    "table.n.02"
                ],
                "attributes": [
                    "white"
                ]
            },
            {
                "object_id": 1060298,
                "x": 18,
                "y": 353,
                "w": 81,
                "h": 44,
                "names": [
                    "handle"
                ],
                "synsets": [
                    "handle.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060299,
                "x": 125,
                "y": 376,
                "w": 141,
                "h": 95,
                "names": [
                    "floor"
                ],
                "synsets": [
                    "floor.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060300,
                "x": 585,
                "y": 169,
                "w": 32,
                "h": 15,
                "names": [
                    "necklace"
                ],
                "synsets": [
                    "necklace.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060245,
                "x": 168,
                "y": 273,
                "w": 175,
                "h": 48,
                "names": [
                    "grey keys"
                ],
                "synsets": [
                    "key.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060301,
                "x": 532,
                "y": 156,
                "w": 52,
                "h": 52,
                "names": [
                    "desk"
                ],
                "synsets": [
                    "desk.n.01"
                ],
                "attributes": None
            },
            {
                "object_id": 1060247,
                "x": 212,
                "y": 340,
                "w": 88,
                "h": 139,
                "names": [
                    "computer tower"
                ],
                "synsets": [],
                "attributes": None
            },
            {
                "object_id": 1060249,
                "x": 2,
                "y": 269,
                "w": 139,
                "h": 201,
                "names": [
                    "cabinet"
                ],
                "synsets": [
                    "cabinet.n.01"
                ],
                "attributes": [
                    "grey",
                    "beige"
                ]
            }
        ],
        "relationships": [
            {
                "relationship_id": 15973,
                "predicate": "in front of",
                "synsets": "['in.r.01']",
                "subject": {
                    "object_id": 1060286,
                    "x": 168,
                    "y": 273,
                    "w": 175,
                    "h": 48,
                    "names": [
                        "keyboard"
                    ],
                    "synsets": [
                        "keyboard.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060246,
                    "x": 159,
                    "y": 119,
                    "w": 149,
                    "h": 155,
                    "names": [
                        "monitor"
                    ],
                    "synsets": [
                        "monitor.n.04"
                    ]
                }
            },
            {
                "relationship_id": 15974,
                "predicate": "under",
                "synsets": "['under.r.01']",
                "subject": {
                    "object_id": 5096,
                    "x": 212,
                    "y": 340,
                    "w": 88,
                    "h": 139,
                    "names": [
                        "cpu"
                    ],
                    "synsets": [
                        "central_processing_unit.n.01"
                    ]
                },
                "object": {
                    "object_id": 5097,
                    "x": 129,
                    "y": 233,
                    "w": 510,
                    "h": 244,
                    "names": [
                        "desktop"
                    ],
                    "synsets": [
                        "desktop.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15975,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 5098,
                    "x": 2,
                    "y": 269,
                    "w": 139,
                    "h": 201,
                    "names": [
                        "filing cabinet"
                    ],
                    "synsets": [
                        "cabinet.n.01"
                    ]
                },
                "object": {
                    "object_id": 5099,
                    "x": 48,
                    "y": 423,
                    "w": 94,
                    "h": 42,
                    "names": [
                        "drawer"
                    ],
                    "synsets": [
                        "drawer.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15976,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 5098,
                    "x": 2,
                    "y": 269,
                    "w": 139,
                    "h": 201,
                    "names": [
                        "filing cabinet"
                    ],
                    "synsets": [
                        "cabinet.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060251,
                    "x": 2,
                    "y": 321,
                    "w": 128,
                    "h": 123,
                    "names": [
                        "drawer"
                    ],
                    "synsets": [
                        "drawer.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15977,
                "predicate": "next to",
                "synsets": "['next.r.01']",
                "subject": {
                    "object_id": 1060253,
                    "x": 346,
                    "y": 296,
                    "w": 29,
                    "h": 28,
                    "names": [
                        "mouse"
                    ],
                    "synsets": [
                        "mouse.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060286,
                    "x": 168,
                    "y": 273,
                    "w": 175,
                    "h": 48,
                    "names": [
                        "keyboard"
                    ],
                    "synsets": [
                        "keyboard.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15978,
                "predicate": "docked in",
                "synsets": "['in.r.01']",
                "subject": {
                    "object_id": 5103,
                    "x": 465,
                    "y": 268,
                    "w": 38,
                    "h": 61,
                    "names": [
                        "wireless phone"
                    ],
                    "synsets": [
                        "telephone.n.01"
                    ]
                },
                "object": {
                    "object_id": 5104,
                    "x": 465,
                    "y": 277,
                    "w": 73,
                    "h": 66,
                    "names": [
                        "office phone base"
                    ],
                    "synsets": []
                }
            },
            {
                "relationship_id": 15979,
                "predicate": "under",
                "synsets": "['under.r.01']",
                "subject": {
                    "object_id": 5096,
                    "x": 212,
                    "y": 340,
                    "w": 88,
                    "h": 139,
                    "names": [
                        "cpu"
                    ],
                    "synsets": [
                        "central_processing_unit.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060282,
                    "x": 130,
                    "y": 235,
                    "w": 506,
                    "h": 241,
                    "names": [
                        "desk"
                    ],
                    "synsets": [
                        "desk.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15980,
                "predicate": "next to",
                "synsets": "['next.r.01']",
                "subject": {
                    "object_id": 5098,
                    "x": 2,
                    "y": 269,
                    "w": 139,
                    "h": 201,
                    "names": [
                        "filing cabinet"
                    ],
                    "synsets": [
                        "cabinet.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060282,
                    "x": 130,
                    "y": 235,
                    "w": 506,
                    "h": 241,
                    "names": [
                        "desk"
                    ],
                    "synsets": [
                        "desk.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15981,
                "predicate": "next to",
                "synsets": "['next.r.01']",
                "subject": {
                    "object_id": 5106,
                    "x": 42,
                    "y": 190,
                    "w": 89,
                    "h": 48,
                    "names": [
                        "multiple outlet"
                    ],
                    "synsets": [
                        "mercantile_establishment.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060246,
                    "x": 159,
                    "y": 119,
                    "w": 149,
                    "h": 155,
                    "names": [
                        "monitor"
                    ],
                    "synsets": [
                        "monitor.n.04"
                    ]
                }
            },
            {
                "relationship_id": 15982,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 5106,
                    "x": 42,
                    "y": 190,
                    "w": 89,
                    "h": 48,
                    "names": [
                        "multiple outlet"
                    ],
                    "synsets": [
                        "mercantile_establishment.n.01"
                    ]
                },
                "object": {
                    "object_id": 5107,
                    "x": 43,
                    "y": 208,
                    "w": 26,
                    "h": 33,
                    "names": [
                        "plug"
                    ],
                    "synsets": [
                        "plug.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15983,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 5106,
                    "x": 42,
                    "y": 190,
                    "w": 89,
                    "h": 48,
                    "names": [
                        "multiple outlet"
                    ],
                    "synsets": [
                        "mercantile_establishment.n.01"
                    ]
                },
                "object": {
                    "object_id": 5108,
                    "x": 66,
                    "y": 207,
                    "w": 15,
                    "h": 21,
                    "names": [
                        "plug"
                    ],
                    "synsets": [
                        "plug.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15984,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 5106,
                    "x": 42,
                    "y": 190,
                    "w": 89,
                    "h": 48,
                    "names": [
                        "multiple outlet"
                    ],
                    "synsets": [
                        "mercantile_establishment.n.01"
                    ]
                },
                "object": {
                    "object_id": 5109,
                    "x": 85,
                    "y": 200,
                    "w": 16,
                    "h": 25,
                    "names": [
                        "plug"
                    ],
                    "synsets": [
                        "plug.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15985,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 5106,
                    "x": 42,
                    "y": 190,
                    "w": 89,
                    "h": 48,
                    "names": [
                        "multiple outlet"
                    ],
                    "synsets": [
                        "mercantile_establishment.n.01"
                    ]
                },
                "object": {
                    "object_id": 5110,
                    "x": 106,
                    "y": 200,
                    "w": 17,
                    "h": 23,
                    "names": [
                        "plug"
                    ],
                    "synsets": [
                        "plug.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15986,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 5091,
                    "x": 456,
                    "y": 266,
                    "w": 98,
                    "h": 79,
                    "names": [
                        "office phone"
                    ],
                    "synsets": []
                },
                "object": {
                    "object_id": 1060282,
                    "x": 130,
                    "y": 235,
                    "w": 506,
                    "h": 241,
                    "names": [
                        "desk"
                    ],
                    "synsets": [
                        "desk.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15987,
                "predicate": "in front of",
                "synsets": "['in.r.01']",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "girl"
                    ],
                    "synsets": [
                        "girl.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060270,
                    "x": 417,
                    "y": 63,
                    "w": 82,
                    "h": 50,
                    "names": [
                        "monitor"
                    ],
                    "synsets": [
                        "monitor.n.04"
                    ]
                }
            },
            {
                "relationship_id": 15988,
                "predicate": "next to",
                "synsets": "['next.r.01']",
                "subject": {
                    "object_id": 1060253,
                    "x": 346,
                    "y": 296,
                    "w": 29,
                    "h": 28,
                    "names": [
                        "mouse"
                    ],
                    "synsets": [
                        "mouse.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060286,
                    "x": 168,
                    "y": 273,
                    "w": 175,
                    "h": 48,
                    "names": [
                        "keyboard"
                    ],
                    "synsets": [
                        "keyboard.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15989,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 5092,
                    "x": 83,
                    "y": 202,
                    "w": 17,
                    "h": 24,
                    "names": [
                        "outlet"
                    ],
                    "synsets": [
                        "mercantile_establishment.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060274,
                    "x": 3,
                    "y": 6,
                    "w": 258,
                    "h": 265,
                    "names": [
                        "wall"
                    ],
                    "synsets": [
                        "wall.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15990,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 5093,
                    "x": 42,
                    "y": 207,
                    "w": 22,
                    "h": 29,
                    "names": [
                        "outlet"
                    ],
                    "synsets": [
                        "mercantile_establishment.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060274,
                    "x": 3,
                    "y": 6,
                    "w": 258,
                    "h": 265,
                    "names": [
                        "wall"
                    ],
                    "synsets": [
                        "wall.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15991,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "girl"
                    ],
                    "synsets": [
                        "girl.n.01"
                    ]
                },
                "object": {
                    "object_id": 5114,
                    "x": 585,
                    "y": 98,
                    "w": 55,
                    "h": 109,
                    "names": [
                        "hair"
                    ],
                    "synsets": [
                        "hair.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15992,
                "predicate": "wears",
                "synsets": "['wear.v.01']",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "girl"
                    ],
                    "synsets": [
                        "girl.n.01"
                    ]
                },
                "object": {
                    "object_id": 5115,
                    "x": 593,
                    "y": 166,
                    "w": 34,
                    "h": 17,
                    "names": [
                        "chain"
                    ],
                    "synsets": [
                        "chain.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15993,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 1060281,
                    "x": 405,
                    "y": 323,
                    "w": 53,
                    "h": 16,
                    "names": [
                        "pen"
                    ],
                    "synsets": [
                        "pen.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060282,
                    "x": 130,
                    "y": 235,
                    "w": 506,
                    "h": 241,
                    "names": [
                        "desk"
                    ],
                    "synsets": [
                        "desk.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15994,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 5117,
                    "x": 137,
                    "y": 408,
                    "w": 52,
                    "h": 58,
                    "names": [
                        "cable"
                    ],
                    "synsets": [
                        "cable.n.01"
                    ]
                },
                "object": {
                    "object_id": 5118,
                    "x": 110,
                    "y": 340,
                    "w": 390,
                    "h": 139,
                    "names": [
                        "floor"
                    ],
                    "synsets": [
                        "floor.n.01"
                    ]
                }
            },
            {
                "relationship_id": 15995,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 5119,
                    "x": 150,
                    "y": 387,
                    "w": 84,
                    "h": 40,
                    "names": [
                        "cable"
                    ],
                    "synsets": [
                        "cable.n.01"
                    ]
                },
                "object": {
                    "object_id": 5118,
                    "x": 110,
                    "y": 340,
                    "w": 390,
                    "h": 139,
                    "names": [
                        "floor"
                    ],
                    "synsets": [
                        "floor.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187822,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 1060274,
                    "x": 3,
                    "y": 6,
                    "w": 258,
                    "h": 265,
                    "names": [
                        "wall"
                    ],
                    "synsets": [
                        "wall.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060261,
                    "x": 42,
                    "y": 190,
                    "w": 89,
                    "h": 48,
                    "names": [
                        "outlets"
                    ],
                    "synsets": [
                        "mercantile_establishment.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187823,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 1060252,
                    "x": 574,
                    "y": 270,
                    "w": 65,
                    "h": 134,
                    "names": [
                        "bag"
                    ],
                    "synsets": [
                        "bag.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060282,
                    "x": 130,
                    "y": 235,
                    "w": 506,
                    "h": 241,
                    "names": [
                        "desk"
                    ],
                    "synsets": [
                        "desk.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187824,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 1060270,
                    "x": 417,
                    "y": 63,
                    "w": 82,
                    "h": 50,
                    "names": [
                        "calendar"
                    ],
                    "synsets": [
                        "calendar.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060287,
                    "x": 304,
                    "y": 30,
                    "w": 217,
                    "h": 97,
                    "names": [
                        "wall"
                    ],
                    "synsets": [
                        "wall.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187825,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "woman"
                    ],
                    "synsets": [
                        "woman.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060277,
                    "x": 585,
                    "y": 98,
                    "w": 55,
                    "h": 109,
                    "names": [
                        "dark hair"
                    ],
                    "synsets": [
                        "hair.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187826,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "woman"
                    ],
                    "synsets": [
                        "woman.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060288,
                    "x": 538,
                    "y": 174,
                    "w": 101,
                    "h": 68,
                    "names": [
                        "white top"
                    ],
                    "synsets": []
                }
            },
            {
                "relationship_id": 3187827,
                "predicate": "with",
                "synsets": "[]",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "woman"
                    ],
                    "synsets": [
                        "woman.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060277,
                    "x": 585,
                    "y": 98,
                    "w": 55,
                    "h": 109,
                    "names": [
                        "dark hair"
                    ],
                    "synsets": [
                        "hair.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187828,
                "predicate": "with",
                "synsets": "[]",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "woman"
                    ],
                    "synsets": [
                        "woman.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060288,
                    "x": 538,
                    "y": 174,
                    "w": 101,
                    "h": 68,
                    "names": [
                        "white t-shirt"
                    ],
                    "synsets": [
                        "jersey.n.03"
                    ]
                }
            },
            {
                "relationship_id": 3187829,
                "predicate": "between",
                "synsets": "['between.r.02']",
                "subject": {
                    "object_id": 1060290,
                    "x": 243,
                    "y": 81,
                    "w": 300,
                    "h": 204,
                    "names": [
                        "partition"
                    ],
                    "synsets": [
                        "partition.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060291,
                    "x": 125,
                    "y": 71,
                    "w": 513,
                    "h": 402,
                    "names": [
                        "cubicles"
                    ],
                    "synsets": [
                        "booth.n.02"
                    ]
                }
            },
            {
                "relationship_id": 3187830,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 1060292,
                    "x": 210,
                    "y": 2,
                    "w": 61,
                    "h": 44,
                    "names": [
                        "picture"
                    ],
                    "synsets": [
                        "picture.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060274,
                    "x": 3,
                    "y": 6,
                    "w": 258,
                    "h": 265,
                    "names": [
                        "wall"
                    ],
                    "synsets": [
                        "wall.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187831,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 1060242,
                    "x": 456,
                    "y": 266,
                    "w": 98,
                    "h": 79,
                    "names": [
                        "telephone"
                    ],
                    "synsets": [
                        "telephone.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060282,
                    "x": 130,
                    "y": 235,
                    "w": 506,
                    "h": 241,
                    "names": [
                        "desk"
                    ],
                    "synsets": [
                        "desk.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187832,
                "predicate": "with",
                "synsets": "[]",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "woman"
                    ],
                    "synsets": [
                        "woman.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060277,
                    "x": 585,
                    "y": 98,
                    "w": 55,
                    "h": 109,
                    "names": [
                        "black hair"
                    ],
                    "synsets": [
                        "hair.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187834,
                "predicate": "next to",
                "synsets": "['next.r.01']",
                "subject": {
                    "object_id": 1060281,
                    "x": 405,
                    "y": 323,
                    "w": 53,
                    "h": 16,
                    "names": [
                        "pen"
                    ],
                    "synsets": [
                        "pen.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060242,
                    "x": 456,
                    "y": 266,
                    "w": 98,
                    "h": 79,
                    "names": [
                        "phone"
                    ],
                    "synsets": [
                        "telephone.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187835,
                "predicate": "working on",
                "synsets": "['work.v.01']",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "woman"
                    ],
                    "synsets": [
                        "woman.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060293,
                    "x": 413,
                    "y": 56,
                    "w": 101,
                    "h": 57,
                    "names": [
                        "computer"
                    ],
                    "synsets": [
                        "computer.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187836,
                "predicate": "are fully",
                "synsets": "['be.v.01']",
                "subject": {
                    "object_id": 1060295,
                    "x": 59,
                    "y": 211,
                    "w": 81,
                    "h": 89,
                    "names": [
                        "cables"
                    ],
                    "synsets": [
                        "cable.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060261,
                    "x": 42,
                    "y": 190,
                    "w": 89,
                    "h": 48,
                    "names": [
                        "plugged"
                    ],
                    "synsets": []
                }
            },
            {
                "relationship_id": 3187837,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 1060296,
                    "x": 209,
                    "y": 3,
                    "w": 341,
                    "h": 45,
                    "names": [
                        "photos"
                    ],
                    "synsets": [
                        "photograph.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060297,
                    "x": 3,
                    "y": 2,
                    "w": 552,
                    "h": 65,
                    "names": [
                        "wall"
                    ],
                    "synsets": [
                        "wall.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187838,
                "predicate": "wear",
                "synsets": "['wear.v.01']",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "woman"
                    ],
                    "synsets": [
                        "woman.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060288,
                    "x": 538,
                    "y": 174,
                    "w": 101,
                    "h": 68,
                    "names": [
                        "white top"
                    ],
                    "synsets": []
                }
            },
            {
                "relationship_id": 3187839,
                "predicate": "have been",
                "synsets": "['be.v.01']",
                "subject": {
                    "object_id": 1060295,
                    "x": 59,
                    "y": 211,
                    "w": 81,
                    "h": 89,
                    "names": [
                        "cables"
                    ],
                    "synsets": [
                        "cable.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060261,
                    "x": 42,
                    "y": 190,
                    "w": 89,
                    "h": 48,
                    "names": [
                        "pluged"
                    ],
                    "synsets": []
                }
            },
            {
                "relationship_id": 3187840,
                "predicate": "has",
                "synsets": "['have.v.01']",
                "subject": {
                    "object_id": 1060251,
                    "x": 2,
                    "y": 321,
                    "w": 128,
                    "h": 123,
                    "names": [
                        "drawer"
                    ],
                    "synsets": [
                        "drawer.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060298,
                    "x": 18,
                    "y": 353,
                    "w": 81,
                    "h": 44,
                    "names": [
                        "handle"
                    ],
                    "synsets": [
                        "handle.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187841,
                "predicate": "WEARING",
                "synsets": "['wear.v.01']",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "woman"
                    ],
                    "synsets": [
                        "woman.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060288,
                    "x": 538,
                    "y": 174,
                    "w": 101,
                    "h": 68,
                    "names": [
                        "white top"
                    ],
                    "synsets": []
                }
            },
            {
                "relationship_id": 3187842,
                "predicate": "next to",
                "synsets": "['next.r.01']",
                "subject": {
                    "object_id": 1060281,
                    "x": 405,
                    "y": 323,
                    "w": 53,
                    "h": 16,
                    "names": [
                        "pen"
                    ],
                    "synsets": [
                        "pen.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060242,
                    "x": 456,
                    "y": 266,
                    "w": 98,
                    "h": 79,
                    "names": [
                        "telephone"
                    ],
                    "synsets": [
                        "telephone.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187843,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 1060295,
                    "x": 59,
                    "y": 211,
                    "w": 81,
                    "h": 89,
                    "names": [
                        "cables"
                    ],
                    "synsets": [
                        "cable.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060299,
                    "x": 125,
                    "y": 376,
                    "w": 141,
                    "h": 95,
                    "names": [
                        "floor"
                    ],
                    "synsets": [
                        "floor.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187844,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 1060294,
                    "x": 421,
                    "y": 105,
                    "w": 99,
                    "h": 95,
                    "names": [
                        "picture"
                    ],
                    "synsets": [
                        "picture.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060290,
                    "x": 243,
                    "y": 81,
                    "w": 300,
                    "h": 204,
                    "names": [
                        "dividing screen"
                    ],
                    "synsets": [
                        "screen.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187845,
                "predicate": "WEARING",
                "synsets": "['wear.v.01']",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "girl"
                    ],
                    "synsets": [
                        "girl.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060300,
                    "x": 585,
                    "y": 169,
                    "w": 32,
                    "h": 15,
                    "names": [
                        "necklace"
                    ],
                    "synsets": [
                        "necklace.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187847,
                "predicate": "with",
                "synsets": "[]",
                "subject": {
                    "object_id": 1060286,
                    "x": 168,
                    "y": 273,
                    "w": 175,
                    "h": 48,
                    "names": [
                        "keyboard"
                    ],
                    "synsets": [
                        "keyboard.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060245,
                    "x": 168,
                    "y": 273,
                    "w": 175,
                    "h": 48,
                    "names": [
                        "grey keys"
                    ],
                    "synsets": [
                        "key.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187848,
                "predicate": "sitting at",
                "synsets": "['sit.v.01']",
                "subject": {
                    "object_id": 1060269,
                    "x": 544,
                    "y": 89,
                    "w": 94,
                    "h": 149,
                    "names": [
                        "woman"
                    ],
                    "synsets": [
                        "woman.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060301,
                    "x": 532,
                    "y": 156,
                    "w": 52,
                    "h": 52,
                    "names": [
                        "desk"
                    ],
                    "synsets": [
                        "desk.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187849,
                "predicate": "under",
                "synsets": "['under.r.01']",
                "subject": {
                    "object_id": 1060247,
                    "x": 212,
                    "y": 340,
                    "w": 88,
                    "h": 139,
                    "names": [
                        "computer tower"
                    ],
                    "synsets": []
                },
                "object": {
                    "object_id": 1060282,
                    "x": 130,
                    "y": 235,
                    "w": 506,
                    "h": 241,
                    "names": [
                        "desk"
                    ],
                    "synsets": [
                        "desk.n.01"
                    ]
                }
            },
            {
                "relationship_id": 3187850,
                "predicate": "ON",
                "synsets": "['along.r.01']",
                "subject": {
                    "object_id": 1060242,
                    "x": 456,
                    "y": 266,
                    "w": 98,
                    "h": 79,
                    "names": [
                        "telephone"
                    ],
                    "synsets": [
                        "telephone.n.01"
                    ]
                },
                "object": {
                    "object_id": 1060248,
                    "x": 129,
                    "y": 233,
                    "w": 510,
                    "h": 244,
                    "names": [
                        "table"
                    ],
                    "synsets": [
                        "table.n.02"
                    ]
                }
            }
        ],
    },
    
]

ICLs = [
    {"role": "user", "content": CONV_PROMPT.format(format_case_vg(icl_cases[0]))},
    {"role": "assistant", "content": "Alright, let's start this conversation. I'm looking at a scene here, and I'm curious about a few things. Can you tell me if about the man in the image wearing a blue hat?"},
    {"role": "user", "content": "Tthere is a person wearing a blue hat on the left of the image by the side of the table"},
    {"role": "assistant", "content": "Interesting! What about any signs or writings in the scene? Can you describe what they say or their colors?"},
    {"role": "user", "content": "In the scene, there is a sign on the wall that says 'no smoking.' the sign is white and placed on the wall, indicating that smoking is not allowed in the area."},
    {"role": "assistant", "content": "Which side of the cake on the table should we cut?"},
    {"role": "user", "content": "There are no cake in the given image."},
    {"role": "assistant", "content": "What is the color of the earphones the woman is wearing in the image?"},
    {"role": "user", "content": "The earphones are in the color of blue."},
    {"role": "assistant", "content": "Where is the television?"},
    {"role": "user", "content": "The television is on the wall in the given image."},
    {"role": "assistant", "content": "Describe the man inside in the image."},
    {"role": "user", "content": "The is no man in the image."},
    {"role": "assistant", "content": "END"}
    ]

print(format_case_vg(icl_cases[0]))


def dyna_conv(args, case, llm_chat):
    conversations = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *ICLs,
                    {"role": "user", "content": CONV_PROMPT.format(format_case_vg(case))}
    ]
    
    to_save = []
    r = 0
    while True:
        message_evaluator = llm_chat.chat(conversations, None)
        
        if "end" in message_evaluator.lower():
            break
        
        conversations.append({"role": "assistant", "content": message_evaluator})
        image_file = case["image"]
        # image_file = Image.open(case["image"]).convert("RGB")
        output = eval_model(model_name, tokenizer, model, image_processor, context_len, type('Args', (), {
                                "model_path": model_path,
                                "model_base": None,
                                "model_name": model_name,
                                "query": message_evaluator,
                                "conv_mode": None,
                                "image_file": image_file,
                                "sep": ",",
                                "load_in_8bit": False,
                                "load_in_4bit": False,
                                "temperature": 0.0,  # set as 0.0 for reproceduce
                                "top_p": None,
                                "num_beams": 1,
                                "max_new_tokens": 512
                            })())
        output = output.lower()
        conversations.append({"role": "user", "content": output})
        print(f"examiner: {message_evaluator}")
        print(f"vlm model: {output}")
        r += 1
        to_save.append(
            {"round_id": r, "prompt": message_evaluator, "response":output}
        )
    return to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--p_mode", type=str, default="certainty")
    parser.add_argument('--model_path', type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument('--outfile', type=str)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    # need to figure out how to eval on different models
    model_name, tokenizer, model, image_processor, context_len = load_model(args.model_path, args.model_base)
    model_path = args.model_path
    samples = load_vg(args.debug)
    # with open('output/vg_samples.json', 'r') as json_file:
    #     samples = json.load(json_file)
    
    llm_chat = LLMChat(model_name="gpt-4o")
    
    print("starting conversation with model...")
    for sample in tqdm.tqdm(samples):
        conv = dyna_conv(args, sample, llm_chat)
        sample["conversations"] = conv
        # print(conv)
        del sample["image"]
    
        with open(args.outfile, "w") as f:
            json.dump(samples, f, indent=4)
