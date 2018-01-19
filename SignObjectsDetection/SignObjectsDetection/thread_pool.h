#pragma once

#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <memory>
#include <condition_variable>

typedef std::function< void() > fn_type;


template< class T >
struct AData
{
	AData() : ready(false) {}
	bool ready;
	T data;
};


class Worker
{
public:

	Worker() :
		mEnabled(true),
		mFqueue(),
		mThread(&Worker::thread_fn, this)
	{}

	~Worker()
	{
		mEnabled = false;
		mCv.notify_one();
		mThread.join();
	}

	void AppendFn(fn_type fn)
	{
		std::unique_lock< std::mutex > locker(mMutex);
		mFqueue.push(fn);
		mCv.notify_one();
	}

	size_t GetTaskCount()
	{
		std::unique_lock< std::mutex > locker(mMutex);
		return mFqueue.size();
	}

	bool IsEmpty()
	{
		std::unique_lock< std::mutex > locker(mMutex);
		return mFqueue.empty();
	}

private:

	bool					mEnabled;
	std::condition_variable mCv;
	std::queue< fn_type >	mFqueue;
	std::mutex				mMutex;
	std::thread				mThread;

	void thread_fn()
	{
		while (mEnabled)
		{
			std::unique_lock< std::mutex > locker(mMutex);
			mCv.wait(locker, [&]() { return !mFqueue.empty() || !mEnabled; });
			while (!mFqueue.empty())
			{
				fn_type fn = mFqueue.front();
				locker.unlock();
				fn();
				locker.lock();
				mFqueue.pop();
			}
		}
	}
};


class ThreadPool
{
public:
	typedef std::shared_ptr< Worker > worker_ptr;

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

	template< class _R, class _FN, class... _ARGS >
	std::shared_ptr< AData< _R > > RunAsync(_FN _fn, _ARGS... _args)
	{
		std::function< _R() > rfn = std::bind(_fn, _args...);
		std::shared_ptr< AData< _R > > p_data(new AData< _R >());
		fn_type fn = [=]()
		{
			p_data->data = rfn();
			p_data->ready = true;
		};
		auto p_worker = GetFreeWorker();
		p_worker->AppendFn(fn);
		return p_data;
	}

	template< class _FN, class... _ARGS >
	void RunAsync(_FN _fn, _ARGS... _args)
	{
		auto p_worker = GetFreeWorker();
		p_worker->AppendFn(std::bind(_fn, _args...));
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

	std::vector< worker_ptr > mWorkers;
};