#pragma once
//C++11∂‡œﬂ≥Ã
#include <thread>
#include <iostream>
#include <mutex>

using namespace std;

typedef enum {
	FrameWorkQueue,
	FrameFreeQueue
} FrameQueueType;

typedef struct FrameQueueNode {
	void    *data;
	size_t  size;  // data size
	long    index;
	struct  FrameQueueNode *next;
} FrameQueueNode;

typedef struct FrameQueue {
	int size;
	FrameQueueType type;
	FrameQueueNode *front;
	FrameQueueNode *rear;
} FrameQueue;

class FrameQueueProcess {

private:
	mutex free_queue_mutex;
	mutex work_queue_mutex;

public:
	FrameQueue *m_free_queue;
	FrameQueue *m_work_queue;

	FrameQueueProcess();
	~FrameQueueProcess();

	// Queue Operation
	void InitQueue(FrameQueue *queue,
		FrameQueueType type);
	void EnQueue(FrameQueue *queue,
		FrameQueueNode *node);
	FrameQueueNode *DeQueue(FrameQueue *queue);
	void ClearFrameQueue(FrameQueue *queue);
	void FreeNode(FrameQueueNode* node);
	void ResetFreeQueue(FrameQueue *workQueue, FrameQueue *FreeQueue);
};