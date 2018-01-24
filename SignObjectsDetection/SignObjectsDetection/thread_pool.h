#pragma once

#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <future>

typedef std::function< void() > func_type;


template< class T >
struct FutureObject
{
	FutureObject() : finished(false) {}
	bool finished;
	T data;
};


class Worker
{
public:

	Worker() :
		mEnabled(true),
		mFqueue(),
		mThread(&Worker::ThreadFunc, this)
	{}

	~Worker()
	{
		mEnabled = false;
		mCv.notify_one();
		mThread.join();
	}

	void AppendFunc(func_type fn);

	size_t GetTaskCount();

	bool IsEmpty();

private:

	bool					mEnabled;
	std::condition_variable mCv;
	std::queue<func_type>	mFqueue;
	std::mutex				mMutex;
	std::thread				mThread;

	void ThreadFunc();
};


class ThreadPool
{
public:
	typedef std::shared_ptr<Worker> worker_ptr;

	std::vector<worker_ptr> mWorkers;

	ThreadPool(size_t threads = 1)
	{
		threads = threads == 0 ? 1 : threads;

		for (size_t i = 0; i < threads; i++)
		{
			worker_ptr p_worker(new Worker);
			mWorkers.push_back(p_worker);
		}
	}

	~ThreadPool() {}

    template< class OBJT, class FUNC, typename... ARGS >
	std::shared_ptr<FutureObject<OBJT>> RunAsync(FUNC function, ARGS... args)
	{
		std::function<OBJT()> rfn = std::bind(function, args...);
		std::shared_ptr<FutureObject<OBJT>> p_data(new FutureObject<OBJT>());
		func_type func = [=]()
		{
			p_data->data = rfn();
			p_data->finished = true;
		};
		auto p_worker = GetFreeWorker();
		p_worker->AppendFunc(func);
		return p_data;
	}

private:
	worker_ptr GetFreeWorker()
	{
		worker_ptr p_worker;
		size_t min_tasks = UINT32_MAX;
		for (auto &it : mWorkers)
		{
			if (it->IsEmpty())
				return it;
			else if (min_tasks > it->GetTaskCount())
			{
				min_tasks = it->GetTaskCount();
				p_worker = it;
			}
		}
		return p_worker;
	}
};