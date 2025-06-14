from rest_framework import serializers
from .models import  History , ChatSession, ChatMessage , Bookmark


class ChatSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatSession
        fields = '__all__'
        # fields = ['session_id', 'user_id', 'created_on']

class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = '__all__'
        # fields = ['session', 'message', 'answer', 'created_on']

# class BookMarkSerializer(serializers.ModelSerializer):
#     # Include input_prompt and ques_id from the related ChatMessage model
#     input_prompt = serializers.CharField(source='message.input_prompt')
#     # ques_id = serializers.CharField(source='message.ques_id')

#     class Meta:
#         model = Bookmark
#         fields = ['id', 'session', 'input_prompt', 'created_on']

class BookMarkSerializer(serializers.ModelSerializer):
    class Meta:
        model = Bookmark
        fields = ['id', 'session', 'created_on']
