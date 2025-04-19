#include "ThreadPool.hpp"

ThreadPool::ThreadPool(size_t num) : is_stop_(false)
{
    for (size_t i = 0; i < num; i++)
    {
        // 初始化线程池
        works_.emplace_back(
            // 定义一个线程日程工作:上锁取任务，通知，做任务
            [&]()
            {
                while (true)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> u_lock(mtx_);
                        cv_.wait(u_lock,
                                 [&]()
                                 {
                                     return !tasks_.empty() || is_stop_;
                                 });
                        if (is_stop_ && tasks_.empty())
                        {
                            return;
                        }

                        // 取任务
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    cv_.notify_one();
                    task();
                }
            });
    }
}

// 此处enqueue默认导入的是 无参、无返回值的 可调用对象
// template <class T>
// void ThreadPool::enqueue(T &&t)
// {
//     {
//         std::unique_lock<std::mutex> u_lock(mtx_);
//         tasks_.emplace(std::forward<T>(t));
//     }
//     cv_.notify_one();
// }

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> u_lock(mtx_);
        is_stop_ = true;
    }
    cv_.notify_all();
    {
        // 使所有work工作线程进入joinable状态
        for (auto &work : works_)
        {
            if (work.joinable())
            {
                work.join();
            }
        }
    }
}
