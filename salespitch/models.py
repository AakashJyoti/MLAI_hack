# Create your models here.
from django.db import models
import uuid
from langchain.schema import SystemMessage, AIMessage, HumanMessage


class ChatSession(models.Model):
    session_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_on = models.DateTimeField(auto_now_add=True)
    user_id = models.CharField(max_length=100, default=" ")
    is_activate = models.BooleanField(default=True)
    session_name = models.CharField(
        max_length=255, blank=True, null=True
    )  # New field for session name

    def __str__(self):
        return f"Chat Session {self.session_id} for User {self.user_id}"


class ChatMessage(models.Model):
    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="messages"
    )
    input_prompt = models.TextField()
    output = models.TextField(blank=True, null=True)
    ques_id = models.TextField(blank=True)  # Unique question ID
    # file_name = models.TextField(blank=True, null=True)
    input_prompt_timestamp = models.TextField(blank=True)
    output_timestamp = models.TextField(blank=True)
    created_on = models.DateTimeField(auto_now_add=True)
    feedback = models.TextField(blank=True, null=True)
    select_feedback_response = models.TextField(blank=True, null=True)
    additional_comments = models.TextField(blank=True, null=True)


class Bookmark(models.Model):
    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="bookmarks"
    )
    message = models.ForeignKey(
        ChatMessage,
        on_delete=models.CASCADE,
        related_name="bookmarks",
        null=True,
        blank=True,
    )
    created_on = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Bookmark for Session {self.session.session_id}, Message ID: {self.message.id}"


class History(models.Model):
    session = models.ForeignKey(
        ChatSession, on_delete=models.CASCADE, related_name="chat_history"
    )
    messages = models.JSONField(default=list)  # Storing chat history as a JSON array
    timestamp = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.phone_number} - {self.timestamp}"

    def get_messages(self):
        # Deserialize JSON messages into LangChain message objects
        return [self.deserialize_message(msg) for msg in self.messages]

    def set_messages(self, messages):
        # Serialize LangChain message objects into JSON-serializable dictionaries
        self.messages = [self.serialize_message(msg) for msg in messages]

    @staticmethod
    def serialize_message(message):
        if isinstance(message, SystemMessage):
            return {"role": "system", "content": message.content}
        elif isinstance(message, AIMessage):
            return {"role": "assistant", "content": message.content}
        elif isinstance(message, HumanMessage):
            return {"role": "user", "content": message.content}
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

    @staticmethod
    def deserialize_message(message):
        role = message["role"]
        content = message["content"]
        if role == "system":
            return SystemMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        elif role == "user":
            return HumanMessage(content=content)
        else:
            raise ValueError(f"Unknown message role: {role}")
