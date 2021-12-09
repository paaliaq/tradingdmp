import asyncio
import logging
import time


async def api_call(id):
    print(f"Send api call {id}")
    await asyncio.sleep(3)
    print(f"Received response api call {id}")
    await inner_api_call(id)
    return 42 + id


async def inner_api_call(id):
    print(f"Send inner api call {id}")
    await asyncio.sleep(3)
    print(f"Received inner response api call {id}")


async def cpu_crunching(id):
    print(f"Start cruning!!! {id}")

    for _ in range(6):
        await asyncio.sleep(0)
        time.sleep(1)

    print(f"End cruning!!! {id}")


async def main():

    tic = time.perf_counter()
    # res_task2 = asyncio.create_task(
    #    cpu_crunching(2)
    # )  # Coroutine / Task / Promise Coroutine<int>
    # res_task1 = asyncio.create_task(
    #    api_call(1)
    # )  # Coroutine / Task / Promise Coroutine<int>

    results = await asyncio.gather(*[api_call(x) for x in range(60)])

    # res1 = await res_task1
    # res2 = await res_task2
    for res in results:
        print(res)

    toc = time.perf_counter()

    # print(res1)
    # print(res2)

    logging.warning(f"Function ran in {toc - tic:0.4f} seconds.")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
