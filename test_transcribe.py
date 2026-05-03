import httpx
import asyncio

async def test():
    async with httpx.AsyncClient() as c:
        try:
            r = await c.post('http://localhost:8001/api/transcribe', files={'audio': ('test.webm', b'fakeaudio', 'audio/webm')})
            print(r.status_code, r.text)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test())
