import asyncio
import threading

def iter_over_async(ait):
    """
    Iterates over an async iterable in a synchronous manner.
    Handles event loop creation, running, and cleanup to avoid "Event loop is closed" errors.
    """
    ait = ait.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    # Get the current event loop or create a new one if it doesn't exist
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:  # No current event loop in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        while True:
            done, obj = loop.run_until_complete(get_next())
            if done:
                break
            yield obj
    finally:
        # Ensure all async tasks are cleaned up before closing
        try:
            pending = asyncio.all_tasks(loop)
            # Gather all pending tasks and wait for them to complete or be cancelled
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except RuntimeError as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Finally, close the loop
            try:
                loop.close()
            except RuntimeError as e:
                print(f"Error closing loop: {e}")
