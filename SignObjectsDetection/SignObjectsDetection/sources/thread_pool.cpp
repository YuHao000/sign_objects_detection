#include "thread_pool.h"


void Worker::AppendFunc(func_type fn)
{
	std::unique_lock<std::mutex> locker(mMutex);
	mFqueue.push(fn);
	mCv.notify_one();
}

size_t Worker::GetTaskCount()
{
	std::unique_lock<std::mutex> locker(mMutex);
	return mFqueue.size();
}

bool Worker::IsEmpty()
{
	std::unique_lock<std::mutex> locker(mMutex);
	return mFqueue.empty();
}

void Worker::ThreadFunc()
{
	while (mEnabled)
	{
		std::unique_lock<std::mutex> locker(mMutex);
		mCv.wait(locker, [&]() { return !mFqueue.empty() || !mEnabled; });
		while (!mFqueue.empty())
		{
			func_type func = mFqueue.front();
			locker.unlock();
			func();
			locker.lock();
			mFqueue.pop();
		}
	}
}