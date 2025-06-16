import logging
from django.http import StreamingHttpResponse
from django.shortcuts import render
import uuid
import asyncio
import re
from langchain_core.messages import SystemMessage, AIMessage
from .serializers import (
    ChatSessionSerializer,
    ChatMessageSerializer,
    BookMarkSerializer,
)
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import get_object_or_404
from .stream_structure_agent8 import ABHFL

# from azure.storage.blob import BlobServiceClient
from .utils import iter_over_async
from .models import ChatSession, ChatMessage, History, Bookmark


logger = logging.getLogger(__name__)


# Render the main page
def my_view(request):
    return render(request, "index.html")


# Utility function to replace slashes with spaces
def replace_slashes(input_string: str) -> str:
    return re.sub(r"[\\/]", " ", input_string)


from threading import Thread


def iter_over_async(ait):
    """
    Iterates over an async iterable in a synchronous manner.
    Uses a queue to bridge the async and sync worlds, avoiding nested event loops.

    Args:
        ait: An asynchronous iterable.

    Yields:
        Each item from the async iterable.
    """
    queue = asyncio.Queue()
    loop = None
    thread = None

    async def producer():
        """Pushes items from the async iterable into the queue."""
        try:
            async for item in ait:
                await queue.put(item)
        finally:
            await queue.put(None)  # Sentinel to signal the end of iteration

    def start_loop():
        """Runs the event loop in a separate thread."""
        nonlocal loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_forever()

    try:
        # Check if an event loop is already running
        try:
            loop = asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False

        if not running:
            # Start a new event loop in a separate thread
            thread = Thread(target=start_loop, daemon=True)
            thread.start()
            while loop is None:  # Wait for the loop to be initialized
                pass

        # Schedule the producer coroutine
        asyncio.run_coroutine_threadsafe(producer(), loop)

        # Consume items from the queue
        while True:
            # Use a thread-safe method to get items from the queue
            future = asyncio.run_coroutine_threadsafe(queue.get(), loop)
            item = future.result()  # Block until the item is available
            if item is None:  # Sentinel value indicates end of iteration
                break
            yield item
    finally:
        if not running and loop is not None:
            # Stop the event loop if we created it
            loop.call_soon_threadsafe(loop.stop)
            # Wait for the loop to stop
            if thread is not None:
                thread.join(timeout=1)  # Wait for the thread to finish
            # Close the loop
            loop.close()


# API to create a new chat session
class NewChatAPIView(generics.CreateAPIView):
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer

    def post(self, request, *args, **kwargs):
        HF_email = request.data.get("HF_email")
        if not HF_email:
            return Response(
                {"error": "HF_email is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        chat_session = ChatSession.objects.create(
            user_id=HF_email, session_id=str(uuid.uuid4())
        )
        return Response(
            self.get_serializer(chat_session).data, status=status.HTTP_201_CREATED
        )


# Main chat handling API
# Optimized ChatAPIView for generating and streaming bot responses without storing data
class ChatAPIView(APIView):
    serializer_class = ChatMessageSerializer

    def post(self, request):
        # Deserialize incoming data
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            session_id = serializer.validated_data.get("session").session_id
            message = serializer.validated_data.get("input_prompt")
            ques_id = serializer.validated_data.get("ques_id")

            # Retrieve the session, if not found return error
            session = get_object_or_404(ChatSession, session_id=session_id)

            # Handle bot interaction if no answer is provided
            chat_history, created = History.objects.get_or_create(session=session)
            # Prepare the history in the desired format
            if created:
                chat_history.messages = []
                # print(history)
            messages = chat_history.get_messages()
            bot_instance = ABHFL(messages)  # Placeholder for bot logic
            response_chunks = []

            
            try:
                    if not bot_instance.message:
                        with open(
                            "prompts/main_prompt2.txt", "r", encoding="utf-8"
                        ) as f:
                            text = f.read()
                        # bot_instance.message.append(("system",text))
                        bot_instance.message.append(SystemMessage(content=text))

                    # asyncio.set_event_loop(loop)
                    questions = replace_slashes(message)
                    
                    response = bot_instance.run_conversation(questions)
                    print(response)
                    response_output = response
                  

                    # final_answer = "".join(response_chunks)
                    final_answer = response_output
                    bot_instance.message.append(AIMessage(content=final_answer))
                    chat_history.set_messages(bot_instance.message)
                    chat_history.save()
                    # yield f"\n[Final Answer Saved for Ques ID: {ques_id}]"
                    
                    return Response({'response': response_output}, status=status.HTTP_200_OK)
            except Exception as e:
                    return Response({'error': f"An error occurred: {e}"}, status=status.HTTP_400_BAD_REQUEST)


# API to store chat messages
# API to store chat messages (for saving/updating user messages)
class StoreChat(APIView):
    serializer_class = ChatMessageSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            session_id = serializer.validated_data.get("session").session_id
            message = serializer.validated_data.get("input_prompt")
            ques_id = serializer.validated_data.get("ques_id")
            answer = serializer.validated_data.get("output", None)
            feedback = serializer.validated_data.get("feedback", None)
            input_prompt_timestamp = serializer.validated_data.get(
                "input_prompt_timestamp", None
            )
            output_timestamp = serializer.validated_data.get("output_timestamp", None)
            feedback = serializer.validated_data.get("feedback", None)
            select_feedback_response = serializer.validated_data.get(
                "select_feedback_response", None
            )
            additional_comments = serializer.validated_data.get(
                "additional_comments", None
            )

            # Validate required fields
            if not all([session_id, message, ques_id]):
                return Response(
                    {
                        "error": "Invalid payload. session_id, message, and ques_id are required."
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Update or create the message in the database
            chat_message, created = ChatMessage.objects.update_or_create(
                session_id=session_id,
                ques_id=ques_id,
                defaults={
                    "input_prompt": message,
                    "output": answer,
                    "input_prompt_timestamp": input_prompt_timestamp,
                    "output_timestamp": output_timestamp,
                    "feedback": feedback,
                    "select_feedback_response": select_feedback_response,
                    "additional_comments": additional_comments,
                },
            )

            status_message = (
                "Message saved successfully"
                if created
                else "Message updated successfully"
            )
            return Response(
                {"status": status_message, "message_id": chat_message.ques_id},
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# History API to retrieve or delete chat session data
class HistoryAPIView(APIView):
    def post(self, request):
        HF_email = request.data.get("HF_email")
        session_id = request.data.get("session", None)

        if HF_email and not session_id:
            sessions = ChatSession.objects.filter(
                user_id=HF_email, is_activate=True
            ).order_by("-created_on")
            response_data = []

            for session in sessions:
                # Retrieve the first message for each session
                first_message = ChatMessage.objects.filter(session=session).first()

                # Build response data for each session
                response_data.append(
                    {
                        "session_id": session.session_id,
                        "first_message": (
                            first_message.input_prompt if first_message else ""
                        ),
                        "session_name": session.session_name,
                        "created_on": first_message.created_on if first_message else "",
                    }
                )

            return Response(response_data, status=status.HTTP_200_OK)

        elif HF_email and session_id:
            try:
                session = ChatSession.objects.get(
                    user_id=HF_email, session_id=session_id
                )
            except ChatSession.DoesNotExist:
                return Response(
                    {"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND
                )

            messages = ChatMessage.objects.filter(session=session).order_by(
                "created_on"
            )

            serializer = ChatMessageSerializer(messages, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response(
            {"error": "Invalid payload"}, status=status.HTTP_400_BAD_REQUEST
        )

    def delete(self, request):
        HF_email = request.data.get("HF_email")
        session_id = request.data.get("session", None)
        print(HF_email, session_id)
        if HF_email and session_id:
            try:
                session = ChatSession.objects.get(
                    user_id=HF_email, session_id=session_id
                )
                session.is_activate = False
                session.save()
                return Response(
                    {"status": "Session deleted successfully"},
                    status=status.HTTP_204_NO_CONTENT,
                )
            except ChatSession.DoesNotExist:
                return Response(
                    {"error": "Session not found"}, status=status.HTTP_404_NOT_FOUND
                )

        return Response(
            {"error": "Invalid payload"}, status=status.HTTP_400_BAD_REQUEST
        )


class BookmarkMessage(APIView):
    def post(self, request):
        """
        Add a bookmark for a session.
        """
        session_id = request.data.get("session_id")

        # Validate the required data
        if not session_id:
            return Response(
                {"detail": "Session_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Fetch the session using session_id
            session = ChatSession.objects.get(session_id=session_id)

            # Check if the session is already bookmarked
            if Bookmark.objects.filter(session=session).exists():
                return Response(
                    {"detail": "This session is already bookmarked."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Create the bookmark
            bookmark = Bookmark.objects.create(session=session)

            # Serialize the bookmark data for response
            serializer = BookMarkSerializer(bookmark)

            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except ChatSession.DoesNotExist:
            return Response(
                {"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND
            )

    def delete(self, request):
        session_id = request.data.get("session_id")

        # Validate the required data
        if not session_id:
            return Response(
                {"detail": "Session_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Fetch the session using session_id
            session = ChatSession.objects.get(session_id=session_id)

            # Check if a bookmark exists for the session
            bookmark = Bookmark.objects.get(session=session)

            # Delete the bookmark
            bookmark.delete()
            return Response(
                {"detail": f"Bookmark deleted successfully for session {session_id}."},
                status=status.HTTP_200_OK,
            )

        except ChatSession.DoesNotExist:
            return Response(
                {"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND
            )
        except Bookmark.DoesNotExist:
            return Response(
                {"detail": "Bookmark not found for this session."},
                status=status.HTTP_404_NOT_FOUND,
            )

    def get(self, request):
        """
        Retrieve details for bookmarked sessions for a specific user_id.
        """
        user_id = request.query_params.get("HF_id")

        # Validate the required data
        if not user_id:
            return Response(
                {"detail": "user_id is required."}, status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Fetch all sessions for the user
            sessions = ChatSession.objects.filter(user_id=user_id)

            # Fetch only sessions that are bookmarked
            bookmarks = Bookmark.objects.filter(session__in=sessions).select_related(
                "session"
            )

            # Prepare response data
            response_data = []
            for bookmark in bookmarks:
                session = bookmark.session
                session = bookmark.session
                first_message = ChatMessage.objects.filter(session=session).first()
                response_data.append(
                    {
                        "session_id": str(session.session_id),
                        "session_name": session.session_name,
                        "first_message": (
                            first_message.input_prompt if first_message else ""
                        ),
                        "created_on": session.created_on,
                    }
                )

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {"detail": f"An error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


# Rename Session View
class RenameSessionAPIView(APIView):

    def patch(self, request):
        """
        Rename an existing chat session.
        """
        session_id = request.data.get("session_id")
        try:
            # Fetch the session
            session = ChatSession.objects.get(session_id=session_id)

            # Get the new name from the request body
            new_name = request.data.get("session_name")

            if not new_name:
                return Response(
                    {"detail": "Session name is required."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Update the session name
            session.session_name = new_name
            session.save()

            # Serialize the updated session data
            serializer = ChatSessionSerializer(session)
            return Response(serializer.data, status=status.HTTP_200_OK)

        except ChatSession.DoesNotExist:
            return Response(
                {"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND
            )
