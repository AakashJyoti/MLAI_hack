from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import get_object_or_404
from .serializers import ChatSessionSerializer, ChatMessageSerializer , BookMarkSerializer
from .models import ChatSession, ChatMessage, History , Bookmark
from django.http import StreamingHttpResponse
from django.shortcuts import render
import uuid
import asyncio
import re
from langchain_core.messages import SystemMessage, AIMessage
from .stream_structure_agent8 import ABHFL
# from azure.storage.blob import BlobServiceClient
import os
import logging
from .utils import iter_over_async
# account_name = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
# account_key = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
logger = logging.getLogger(__name__)
# Render the main page
def my_view(request):
    return render(request, "index.html")


# Utility function to replace slashes with spaces
def replace_slashes(input_string: str) -> str:
    return re.sub(r'[\\/]', ' ', input_string)


# Utility function to iterate over async generator
# def iter_over_async(ait):
#     import threading
#     ait = ait.__aiter__()

#     async def get_next():
#         try:
#             obj = await ait.__anext__()
#             return False, obj
#         except StopAsyncIteration:
#             return True, None

#     # Ensure an event loop exists in the current thread
#     try:
#         loop = asyncio.get_event_loop()
#     except RuntimeError:  # No current event loop in this thread
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#     try:
#         while True:
#             done, obj = loop.run_until_complete(get_next())
#             if done:
#                 break
#             yield obj
#     # finally:
#     #     loop.close()
#     finally:
#         # Ensure all async tasks are cleaned up before closing
#         pending = asyncio.all_tasks(loop)
#         for task in pending:
#             task.cancel()
#         loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
#         loop.close()

from threading import Thread
# def iter_over_async(ait):
#     """
#     Iterates over an async iterable in a synchronous manner.
#     """
#     ait = ait.__aiter__()

#     async def get_next():
#         try:
#             obj = await ait.__anext__()
#             return False, obj
#         except StopAsyncIteration:
#             return True, None

#     try:
#         loop = asyncio.get_running_loop()  # This will raise RuntimeError if no loop is running
#         running = True
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
#         running = False

#     try:
#         while True:
#             if running:
#                 # Use thread-safe method if loop is already running
#                 future = asyncio.run_coroutine_threadsafe(get_next(), loop)
#                 done, obj = future.result()
#             else:
#                 done, obj = loop.run_until_complete(get_next())
#             if done:
#                 break
#             yield obj
#     finally:
#         # If we created the loop, close it; otherwise, leave it running.
#         if not running:
#             try:
#                 pending = asyncio.all_tasks(loop)
#                 if pending:
#                     loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
#             except RuntimeError as e:
#                 print(f"Error during cleanup: {e}")
#             finally:
#                 try:
#                     loop.close()
#                 except RuntimeError as e:
#                     print(f"Error closing loop: {e}")

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
        HF_email = request.data.get('HF_email')
        if not HF_email:
            return Response({"error": "HF_email is required"}, status=status.HTTP_400_BAD_REQUEST)

        chat_session = ChatSession.objects.create(user_id=HF_email, session_id=str(uuid.uuid4()))
        return Response(self.get_serializer(chat_session).data, status=status.HTTP_201_CREATED)


# Main chat handling API
# Optimized ChatAPIView for generating and streaming bot responses without storing data
class ChatAPIView(APIView):
    serializer_class = ChatMessageSerializer

    def post(self, request):
        # Deserialize incoming data
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            session_id = serializer.validated_data.get('session').session_id
            message = serializer.validated_data.get('input_prompt')
            ques_id = serializer.validated_data.get('ques_id')
            
            # Retrieve the session, if not found return error
            session = get_object_or_404(ChatSession, session_id=session_id)

            # Handle bot interaction if no answer is provided
            chat_history, created = History.objects.get_or_create(session = session)
                # Prepare the history in the desired format
            if created:
                chat_history.messages = []
                # print(history)
            messages = chat_history.get_messages()
            bot_instance = ABHFL(messages)  # Placeholder for bot logic
            response_chunks = []

            def generate():
                try:
                    if not bot_instance.message:
                        with open("prompts/main_prompt2.txt", "r", encoding="utf-8") as f:
                            text = f.read()
                        bot_instance.message.append(SystemMessage(content=text))
                    
                    
                    # asyncio.set_event_loop(loop)
                    questions = replace_slashes(message)
                    openai_response = iter_over_async(bot_instance.run_conversation(questions.lower()))

                    for event in openai_response:
                        kind = event["event"]
                        if kind == "on_chat_model_stream":
                            content = event["data"]["chunk"].content
                            if content:
                                response_chunks.append(content)
                                yield content

                    final_answer = "".join(response_chunks)
                    bot_instance.message.append(AIMessage(content=final_answer))
                    chat_history.set_messages(bot_instance.message)
                    chat_history.save()
                    # yield f"\n[Final Answer Saved for Ques ID: {ques_id}]"

                except Exception as e:
                    yield f"Oops, something went wrong processing your request. Please try again."

            response = StreamingHttpResponse(generate(), content_type="text/event-stream")
            response["Cache-Control"] = "no-cache"
            response["X-Accel-Buffering"] = "no"
            return response


        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



# API to store chat messages
# API to store chat messages (for saving/updating user messages)
class StoreChat(APIView):
    serializer_class = ChatMessageSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            session_id = serializer.validated_data.get('session').session_id
            message = serializer.validated_data.get('input_prompt')
            ques_id = serializer.validated_data.get('ques_id')
            answer = serializer.validated_data.get('output', None)
            feedback = serializer.validated_data.get('feedback', None)
            input_prompt_timestamp = serializer.validated_data.get('input_prompt_timestamp', None)
            output_timestamp = serializer.validated_data.get('output_timestamp', None)
            feedback = serializer.validated_data.get('feedback', None)
            select_feedback_response = serializer.validated_data.get('select_feedback_response', None)
            additional_comments = serializer.validated_data.get('additional_comments', None)

            # Validate required fields
            if not all([session_id, message, ques_id]):
                return Response({'error': 'Invalid payload. session_id, message, and ques_id are required.'}, status=status.HTTP_400_BAD_REQUEST)

            # Update or create the message in the database
            chat_message, created = ChatMessage.objects.update_or_create(
                session_id=session_id,
                ques_id=ques_id,
                defaults={
                'input_prompt': message,
                'output': answer,
                'input_prompt_timestamp': input_prompt_timestamp,
                'output_timestamp': output_timestamp,
                'feedback': feedback,
                'select_feedback_response': select_feedback_response,
                'additional_comments': additional_comments
            }
            )

            status_message = 'Message saved successfully' if created else 'Message updated successfully'
            return Response({
                'status': status_message,
                'message_id': chat_message.ques_id
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# History API to retrieve or delete chat session data
class HistoryAPIView(APIView):
    def post(self, request):
        HF_email = request.data.get('HF_email')
        session_id = request.data.get('session', None)

        if HF_email and not session_id:
            sessions = ChatSession.objects.filter(user_id=HF_email, is_activate=True).order_by("-created_on")
            response_data = []

            for session in sessions:
                # Retrieve the first message for each session
                first_message = ChatMessage.objects.filter(session=session).first()
                
                # Build response data for each session
                response_data.append({
                    'session_id': session.session_id,
                    'first_message': first_message.input_prompt if first_message else '',
                    'session_name' : session.session_name,
                    'created_on': first_message.created_on if first_message else ""
                })

            return Response(response_data, status=status.HTTP_200_OK)

        elif HF_email and session_id:
            try:
                session = ChatSession.objects.get(user_id=HF_email, session_id=session_id)
            except ChatSession.DoesNotExist:
                return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)

            messages = ChatMessage.objects.filter(session=session).order_by('created_on')
            
            serializer = ChatMessageSerializer(messages, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response({'error': 'Invalid payload'}, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request):
        HF_email = request.data.get('HF_email')
        session_id = request.data.get('session', None)
        print(HF_email , session_id)
        if HF_email and session_id:
            try:
                session = ChatSession.objects.get(user_id=HF_email, session_id=session_id)
                session.is_activate = False
                session.save()
                return Response({'status': 'Session deleted successfully'}, status=status.HTTP_204_NO_CONTENT)
            except ChatSession.DoesNotExist:
                return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)

        return Response({'error': 'Invalid payload'}, status=status.HTTP_400_BAD_REQUEST)

# Bookmark View
# class BookmarkMessage(APIView):
#     def post(self, request):
#         """
#         Add a bookmark for a specific message in a session using ques_id.
#         """
#         session_id = request.data.get('session_id')
#         ques_id = request.data.get('ques_id')

#         # Validate the required data
#         if not session_id or not ques_id:
#             return Response({"detail": "Both session_id and ques_id are required."}, status=status.HTTP_400_BAD_REQUEST)
        
#         try:
#             # Fetch the session using session_id
#             session = ChatSession.objects.get(session_id=session_id)
            
#             # Fetch the message using ques_id and session
#             # message = ChatMessage.objects.get(ques_id=ques_id, session=session)
            
#             # Check if the message is already bookmarked
#             if Bookmark.objects.filter(session=session).exists():
#                 return Response({"detail": "This message is already bookmarked."}, status=status.HTTP_400_BAD_REQUEST)
            
#             # Create the bookmark
#             bookmark = Bookmark.objects.create(session=session)
            
#             # Serialize the bookmark data for response
#             serializer = BookMarkSerializer(bookmark)
#             print(serializer.data)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)

#         except ChatSession.DoesNotExist:
#             return Response({"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
#         except ChatMessage.DoesNotExist:
#             return Response({"detail": "Message not found."}, status=status.HTTP_404_NOT_FOUND)

#     def delete(self , request):
#         session_id = request.data.get('session_id')
#         ques_id = request.data.get('ques_id')


#         # Validate the required data
#         if not session_id or not ques_id:
#             return Response({"detail": "Both session_id and ques_id are required."}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             # Fetch the session using session_id
#             session = ChatSession.objects.get(session_id=session_id)
            
#             # Fetch the message using ques_id and session
#             message = ChatMessage.objects.get(ques_id=ques_id, session=session)

#              # Create the bookmark
#             bookmark = Bookmark.objects.get(session=session, message=message)
            
#             bookmark.delete()
#             return Response({"detail": f"BookMark Deleted succesfully. for {session_id} and {ques_id}"},status=status.HTTP_200_OK)

#         except ChatSession.DoesNotExist:
#             return Response({"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
#         except ChatMessage.DoesNotExist:
#             return Response({"detail": "Message not found."}, status=status.HTTP_404_NOT_FOUND)

#     def get(self, request):
#         """
#         Retrieve all bookmarks for a specific user_id.
#         """
#         user_id = request.query_params.get('HF_id')

#         # Validate the required data
#         if not user_id:
#             return Response({"detail": "user_id is required."}, status=status.HTTP_400_BAD_REQUEST)

#         # Fetch all sessions for the user
#         sessions = ChatSession.objects.filter(user_id=user_id)

#         # Fetch all bookmarks for the user's sessions
#         bookmarks = Bookmark.objects.filter(session__in=sessions)

#         # Prepare response data
#         response_data = {}
#         for bookmark in bookmarks:
#             session_id = str(bookmark.session.session_id)
#             if session_id not in response_data:
#                 response_data[session_id] = {
#                     "session_name": session_id,
#                     "created_on": bookmark.session.created_on,
#                     "bookmarks": []
#                 }
            
#             response_data[session_id]["bookmarks"].append({
#                 "ques_id": bookmark.message.ques_id,
#                 "input_prompt": bookmark.message.input_prompt,
#                 "output": bookmark.message.output,
#                 "created_on": bookmark.message.created_on,
#                 "feedback": bookmark.message.feedback,
#                 "additional_comments": bookmark.message.additional_comments,
#             })

#         return Response(response_data.values(), status=status.HTTP_200_OK)

class BookmarkMessage(APIView):
    def post(self, request):
        """
        Add a bookmark for a session.
        """
        session_id = request.data.get('session_id')

        # Validate the required data
        if not session_id:
            return Response({"detail": "Session_id is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Fetch the session using session_id
            session = ChatSession.objects.get(session_id=session_id)
            
            # Check if the session is already bookmarked
            if Bookmark.objects.filter(session=session).exists():
                return Response({"detail": "This session is already bookmarked."}, status=status.HTTP_400_BAD_REQUEST)
            
            # Create the bookmark
            bookmark = Bookmark.objects.create(session=session)
            
            # Serialize the bookmark data for response
            serializer = BookMarkSerializer(bookmark)
            
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        except ChatSession.DoesNotExist:
            return Response({"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        
    def delete(self, request):
        session_id = request.data.get('session_id')

        # Validate the required data
        if not session_id:
            return Response({"detail": "Session_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Fetch the session using session_id
            session = ChatSession.objects.get(session_id=session_id)

            # Check if a bookmark exists for the session
            bookmark = Bookmark.objects.get(session=session)
            
            # Delete the bookmark
            bookmark.delete()
            return Response({"detail": f"Bookmark deleted successfully for session {session_id}."}, status=status.HTTP_200_OK)

        except ChatSession.DoesNotExist:
            return Response({"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        except Bookmark.DoesNotExist:
            return Response({"detail": "Bookmark not found for this session."}, status=status.HTTP_404_NOT_FOUND)

    def get(self, request):
        """
        Retrieve details for bookmarked sessions for a specific user_id.
        """
        user_id = request.query_params.get('HF_id')

        # Validate the required data
        if not user_id:
            return Response({"detail": "user_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Fetch all sessions for the user
            sessions = ChatSession.objects.filter(user_id=user_id)

            # Fetch only sessions that are bookmarked
            bookmarks = Bookmark.objects.filter(session__in=sessions).select_related('session')
            
            # Prepare response data
            response_data = []
            for bookmark in bookmarks:
                session = bookmark.session
                session = bookmark.session
                first_message = ChatMessage.objects.filter(session=session).first()
                response_data.append({
                    "session_id": str(session.session_id),
                    "session_name": session.session_name,
                    "first_message": first_message.input_prompt if first_message else '',
                    "created_on": session.created_on,
                })

            return Response(response_data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"detail": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Rename Session View
class RenameSessionAPIView(APIView):
    
    def patch(self, request):
        """
        Rename an existing chat session.
        """
        session_id = request.data.get('session_id')
        try:
            # Fetch the session
            session = ChatSession.objects.get(session_id=session_id)

            # Get the new name from the request body
            new_name = request.data.get("session_name")

            if not new_name:
                return Response({"detail": "Session name is required."}, status=status.HTTP_400_BAD_REQUEST)
            
            # Update the session name
            session.session_name = new_name
            session.save()

            # Serialize the updated session data
            serializer = ChatSessionSerializer(session)
            return Response(serializer.data, status=status.HTTP_200_OK)

        except ChatSession.DoesNotExist:
            return Response({"detail": "Session not found."}, status=status.HTTP_404_NOT_FOUND)

# class FileProcessingAPIView(APIView):

#     def post(self, request):
#         serializer = self.serializer_class(data=request.data)
#         if not serializer.is_valid():
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#         # Extract validated data
#         data = serializer.validated_data
#         session_id = data.get('session').session_id
#         message = data.get('input_prompt')
#         ques_id = data.get('ques_id')
#         file_name = data.get('file_name')

#         # Retrieve session and chat history
#         session = get_object_or_404(ChatSession, session_id=session_id)
#         chat_history = self.get_or_create_chat_history(session)

#         # Handle missing file name
#         if not file_name:
#             return Response({"error": "No file URL provided."}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             # Download file content
#             file_content = self.download_blob_content(file_name)

#             # Process file content and generate response
#             response = self.stream_response(file_content, message, chat_history, ques_id)
#             return response

#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#     def get_or_create_chat_history(self, session):
#         chat_history, created = History.objects.get_or_create(session=session)
#         if created:
#             chat_history.messages = []
#         return chat_history

#     def download_blob_content(self, file_name):
#         connection_string = (f"DefaultEndpointsProtocol=https;"
#                              f"AccountName={account_name};"
#                              f"AccountKey={account_key};"
#                              f"EndpointSuffix=core.windows.net")
#         blob_service_client = BlobServiceClient.from_connection_string(connection_string)
#         container_client = blob_service_client.get_container_client('mastertablecsb')
#         blob_client = container_client.get_blob_client(file_name)
#         blob_stream = blob_client.download_blob()
#         return blob_stream.readall().decode("utf-8")

#     def stream_response(self, file_content, message, chat_history, ques_id):
#         def generate():
#             if not chat_history.get_messages():
#                 bot_instance = ABHFL([])  # Initialize bot with an empty history
#             else:
#                 bot_instance = ABHFL(chat_history.get_messages())  # Initialize bot with chat history

#             bot_instance.message.append(SystemMessage(content=file_content))
#             questions = replace_slashes(message)
#             response_chunks = []

#             try:
#                 openai_response = iter_over_async(bot_instance.run_conversation(questions.lower()))
#                 for event in openai_response:
#                     if event["event"] == "on_chat_model_stream":
#                         content = event["data"]["chunk"].content
#                         if content:
#                             response_chunks.append(content)
#                             yield content

#                 final_answer = "".join(response_chunks)
#                 bot_instance.message.append(AIMessage(content=final_answer))
#                 chat_history.set_messages(bot_instance.message)
#                 chat_history.save()

#             except Exception as e:
#                 yield f"Error: {str(e)}"

#         response = StreamingHttpResponse(generate(), content_type="text/event-stream")
#         response["Cache-Control"] = "no-cache"
#         response["X-Accel-Buffering"] = "no"
#         return response


# class ImageProcessing():
