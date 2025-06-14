from django.urls import path
from .views import (
    my_view,
    HistoryAPIView,
    NewChatAPIView,
    ChatAPIView,
    StoreChat,
    BookmarkMessage,
    RenameSessionAPIView,
)

urlpatterns = [
    path("", my_view, name="my_view"),
    path("history/", HistoryAPIView.as_view(), name="history"),
    path("new_chat/", NewChatAPIView.as_view(), name="new_chat"),
    path("chat/", ChatAPIView.as_view(), name="chat"),
    path("store/", StoreChat.as_view(), name="store_chat"),
    path("bookmark/", BookmarkMessage.as_view(), name="bookmark"),
    path("rename_session/", RenameSessionAPIView.as_view(), name="rename_session"),
    # path('file_processing/', FileProcessingAPIView.as_view(), name='file_processing'),
]
