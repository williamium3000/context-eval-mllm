CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""
CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY_COVERAGE = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
6. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_COCO_ICL_CERTAINTY_COVERAGE = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
6. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_CERTAINTY = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_COCO_ICL_CERTAINTY = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_COVERAGE = \
"""
Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
You need to have multiple conversations the model over multiple images. Each conversation will be based on one single image. The given image will be presented to you as a list of objects with attributes and relation of these objects. Each objects will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
Given this image, you need to perform a series of casual conversations with the vision-language model naturally and ask questions about the given detailed information of the image.
The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

You will be given multiple images, and you need to have multiple conversations with the vision-language model. Once you are given the information of the image, you should start the conversation about this image. 
"""
CONV_MODEL_PERSPECTIVE_PROMPT_VG_ICL_UNANSWERABLE = \
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
CONV_MODEL_PERSPECTIVE_PROMPT_COCO_ICL_UNANSWERABLE = \
"""
Task: Your task is to have multiple conversations with a vision-language model. Each conversation will be based on one single image. The image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
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

CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE = \
"""Task: Your task is to have a multi-round conversation with a vision-language model regarding a given image. The image will be presented to you as a list of instances with attributes and relation of these instances. Each instance will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_UNANSWERABLE = \
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
"""

CONV_MODEL_PERSPECTIVE_PROMPT_COCO_UNANSWERABLE = \
"""
Task: Your task is to have multiple conversations with a vision-language model. Each conversation will be based on one single image. The image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
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
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATE = \
"""Task: Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
The image will be presented to you as a list of instances with attributes and relation of these instances. Each instance will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_COCO_SIMGPLE_INTERROGATE = \
"""Task: Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
The image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATEV2 = \
"""Task: Your task is to evaluate whether a vision-language model is hallucinating (i.e. generate responses contradictory to the content of the image) on the image.
The image will be presented to you as a list of instances with attributes and relation of these instances. Each instance will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You should perform a series of casual conversations with the model naturally by asking questions or making statements about the image and determine whether the model is hallucinating or not in its response.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATEV2_NO_COVERAGE = \
"""Task: Your task is to evaluate whether a vision-language model is hallucinating (i.e. generate responses contradictory to the content of the image) on the image.
The image will be presented to you as a list of instances with attributes and relation of these instances. Each instance will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You should perform a series of casual conversations with the model naturally by asking questions or making statements about the image and determine whether the model is hallucinating or not in its response.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""

CONV_MODEL_PERSPECTIVE_PROMPT_VG_SIMGPLE_INTERROGATE_CERTAINTY = \
"""Task: Your task is to have a multi-round conversation with a vision-language model regarding a given image. The responses from the vision-language model will be later to used to evaluate whether the model is hallucinating or faithful to the image (wheteher model generate responses contradictory to the content of the image).
The image will be presented to you as a list of instances with attributes and relation of these instances. Each instance will be presented with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.

Requirements:
1. The conversation is multi-turn and open-ended. you need to ask questions based on both the image content and the history of the conversations. At each round of the conversation, you should only provide your part of the conversation and wait for the human to respond. 
2. You should make the conversation as natural as possible. Act as if you are a human having causal conversation directly with another human.
3. Do NOT correct the model if it makes mistakes. However, you can ask further regarding the mistakes to confirm or interrogate the model. However, DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention anything about the information source, e.g. bounding box.
4. The whole conversation should COVER all the information regarding the image. If the human responses fail to cover some specific object, attributes or relations in the image, you should ask about it in the subsequent conversations.
5. Ask diverse questions. DO NOT ask any question that cannot be answered from the given image information confidently. ONLY include questions that have definite answers
    a. one can see the content in the image that the question asks about and can answer confidently.
    b. one can determine confidently from the image that it is not in the image.
6. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""


CONV_COVERAGE_PROMPT = \
"""
Task: Your task is to have a conversation with a human regarding the image provided to you. This image will be presented to you as five captions, each describing the same image you are observing, and a list of objects with specific coordinates locations in the image, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y.
You need to perform a series of casual conversations with a human naturally by asking questions or making statements about the image to assess human's perception of the image (whether the human will hallucinate, provide information contradictory to the image or fail to have knowledge regarding some parts or entities in the image).
The conversation is multi-turn and can be open-ended and you need to ask questions based on the history of the conversations. 

Requirements:
1. The conversation is a multi-turn process and your current response should be based on the history of the conversations.
2. At each round, you should only provide your part of the conversation and wait for the human to respond.
3. You should make the conversation as natural as possible and act as if you are a human having conversation directly with another human.
4. COVERAGE: the whole conversation is expected to COVER all the information regarding the image, you should ask questions that cover as many details as possible. If the human responses fail to cover some specific object, attributes or relations in the image, you should cover this in the subsequent conversations.
5. DO NOT DISCLOSE any given image information (captions and bboxes) directly to the human in your conversation. Also, DO NOT mention that the information source is the caption or the bounding box.
6. Focus your conversation on the content of the image. Remember the purpose of the conversation is to assess the human's perception and knowledge of the image.
7. You can end the conversation naturally. If you feel like the conversation is coming to an end, you can end the conversation by outputing "END" ONLY. 

Bad examples:
Image information:
"A man is in a kitchen making pizzas.",
"Man in apron standing on front of oven with pans and bakeware",
"A baker is working in the kitchen rolling dough.",
"A person standing by a stove in a kitchen.",
"A table with pies being made and a person standing near a wall with pots and pans hanging on the wall."

You: Are there any smaller items, like a bottle or specific ingredient like broccoli or carrots, you can identify in the scene?
Reason: you should not mention the objects in the image directly. For example, you can ask: Are there any smaller items you can identify in the scene?

You: The organization definitely adds to the efficiency. I also noticed a bottle on the table. What do you think it could contain\u2014perhaps some oil, vinegar, or maybe a seasoning for the dishes?
Reason: This question cannot be answered confidently from the given image information. You should only ask questions that have definite answers.

You: What do you think about the rack or wall with pots and pans hanging around the workspace? Does it contribute to the vibe of the kitchen?
Reason: first, do not mention detailed information from the image directly. Second, this question cannot be answered confidently from the given image information. You should only ask questions that have definite answers.


Image information:
{}

Please respond as if you are having the conversation with the vision-language model directly.
"""
