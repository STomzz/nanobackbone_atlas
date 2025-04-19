#include "ThreadPool.hpp"

int main()
{
    std::mutex common_num_mutex;
    std::condition_variable common_num_cv;
    std::unordered_map<std::string, int> mymap;
    mymap["common_num"] = 0;
    ThreadPool pool(2);
    pool.enqueue(
        [&mymap, &common_num_mutex, &common_num_cv]()
        {
            while (true)
            {
                {
                    std::unique_lock<std::mutex> u_lock(common_num_mutex);
                    common_num_cv.wait(u_lock,
                                       [&]()
                                       {
                                           return mymap["common_num"] == 0;
                                       });
                    mymap["common_num"]++;
                    printf("common_num++ now is : %d\n", mymap["common_num"]);
                }
                common_num_cv.notify_one();
            }
        });
    pool.enqueue(
        [&mymap, &common_num_mutex, &common_num_cv]()
        {
            while (true)
            {
                {
                    std::unique_lock<std::mutex> u_lock(common_num_mutex);
                    common_num_cv.wait(u_lock,
                                       [&]()
                                       {
                                           return mymap["common_num"] == 1;
                                       });
                    mymap["common_num"]--;
                    printf("common_num-- now is : %d\n", mymap["common_num"]);
                }
                common_num_cv.notify_one();
            }
        });

    return 0;
}