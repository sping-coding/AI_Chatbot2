from django.shortcuts import render
from django.http import JsonResponse
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch
from django.views.decorators.csrf import csrf_exempt
from .models import Conversation  # 추가

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PreTrainedTokenizerFast.from_pretrained('byeongal/Ko-DialoGPT')
model = GPT2LMHeadModel.from_pretrained('byeongal/Ko-DialoGPT').to(device)

@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        text_idx = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        inference_output = model.generate(
            text_idx,
            max_length=1000,
            num_beams=5,
            top_k=20,
            no_repeat_ngram_size=4,
            length_penalty=0.65,
            repetition_penalty=2.0,
        )
        inference_output = inference_output.tolist()
        bot_response = tokenizer.decode(inference_output[0][text_idx.shape[-1]:], skip_special_tokens=True)

        # 대화 내용을 데이터베이스에 저장
        conversation = Conversation(user_input=user_input, bot_response=bot_response)
        conversation.save()

        return JsonResponse({'bot_response': bot_response})
    
    # 대화 내용을 가져와 템플릿에 전달
    conversation_history = Conversation.objects.all()
    return render(request, 'chatbot_app/chatbot.html', {'conversation_history': conversation_history})
