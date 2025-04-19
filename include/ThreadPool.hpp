#pragma once

#include <thread>
#include <iostream>
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>
#include <functional>

class ThreadPool
{
public:
    explicit ThreadPool(size_t num);
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    template <class T>
    void enqueue(T &&t)
    {
        {
            std::unique_lock<std::mutex> u_lock(mtx_);
            tasks_.emplace(std::forward<T>(t));
        }
        cv_.notify_one();
    }

    ~ThreadPool();

private:
    bool is_stop_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<std::thread> works_;
    std::queue<std::function<void()>> tasks_;
};