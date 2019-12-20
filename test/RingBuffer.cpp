/*******************************************************************************************************************
本例需求：将相机回调产生的数据放入高效的队列中，然后异步开启一条线程专门处理从相机回调装入队列的Buffer，实现高效处理每一帧数据(如进行crop,scale...)
验证：可查看Log,如果控制台打印的Test : index 是连续的，则说明装入队列是按顺序的成功。
注意：关于C++队列实现可直接在FrameQueueProcess.h FrameQueueProcess.mm文件中直接看到。
具体详细解析请参考简书或者博客，如果喜欢记得在GitHub里， 简书里给个星星，给个赞，Thanks
简书地址：
博客地址：
GitHub地址：https://github.com/ChengyangLi/WorkQueueAndFreeQueue_HandleData
********************************************************************************************************************/
#include "RingBuffer.h"

const int FrameQueueSize = 3;

const static char *kModuleName = "FrameQueueProcess";

#pragma mark - Init
FrameQueueProcess::FrameQueueProcess() {
	m_free_queue = (FrameQueue *)malloc(sizeof(struct FrameQueue));
	m_work_queue = (FrameQueue *)malloc(sizeof(struct FrameQueue));

	InitQueue(m_free_queue, FrameFreeQueue);
	InitQueue(m_work_queue, FrameWorkQueue);

	for (int i = 0; i < FrameQueueSize; i++) {
		FrameQueueNode *node = (FrameQueueNode *)malloc(sizeof(struct FrameQueueNode));
		node->data = NULL;
		node->size = 0;
		node->index = 0;
		this->EnQueue(m_free_queue, node);
	}

	//pthread_mutex_init(&free_queue_mutex, NULL);
	//pthread_mutex_init(&work_queue_mutex, NULL);

	//log4cplus_info(kModuleName, "%s: Init finish !", __func__);
}

void FrameQueueProcess::InitQueue(FrameQueue *queue, FrameQueueType type) {
	if (queue != NULL) {
		queue->type = type;
		queue->size = 0;
		queue->front = 0;
		queue->rear = 0;
	}
}

#pragma mark - Main Operation
void FrameQueueProcess::EnQueue(FrameQueue *queue, FrameQueueNode *node) {
	if (queue == NULL) {
		//log4cplus_debug(kModuleName, "%s: current queue is NULL", __func__);
		return;
	}

	if (node == NULL) {
		//log4cplus_debug(kModuleName, "%s: current node is NUL", __func__);
		return;
	}

	node->next = NULL;

	if (FrameFreeQueue == queue->type) {
		//pthread_mutex_lock(&free_queue_mutex);
		free_queue_mutex.lock();
		if (queue->front == NULL) {
			queue->front = node;
			queue->rear = node;
		}
		else {
			/*
			// tail in,head out
			freeQueue->rear->next = node;
			freeQueue->rear = node;
			*/

			// head in,head out
			node->next = queue->front;
			queue->front = node;
		}
		queue->size += 1;
		//log4cplus_debug(kModuleName, "%s: free queue size=%d", __func__, queue->size);
		//pthread_mutex_unlock(&free_queue_mutex);
		free_queue_mutex.unlock();
	}

	if (FrameWorkQueue == queue->type) {
		//pthread_mutex_lock(&work_queue_mutex);
		work_queue_mutex.lock();
		//TODO
		static long nodeIndex = 0;
		node->index = (++nodeIndex);
		if (queue->front == NULL) {
			queue->front = node;
			queue->rear = node;
		}
		else {
			queue->rear->next = node;
			queue->rear = node;
		}
		queue->size += 1;
		//log4cplus_debug(kModuleName, "%s: work queue size=%d", __func__, queue->size);
		//pthread_mutex_unlock(&work_queue_mutex);
		work_queue_mutex.unlock();
	}
}

FrameQueueNode* FrameQueueProcess::DeQueue(FrameQueue *queue) {
	if (queue == NULL) {
		//log4cplus_debug(kModuleName, "%s: current queue is NULL", __func__);
		return NULL;
	}

	const char *type = queue->type == FrameWorkQueue ? "work queue" : "free queue";
	mutex *queue_mutex = ((queue->type == FrameWorkQueue) ? &work_queue_mutex : &free_queue_mutex);
	FrameQueueNode *element = NULL;

	//pthread_mutex_lock(queue_mutex);
	queue_mutex->lock();
	element = queue->front;
	if (element == NULL) {
		//pthread_mutex_unlock(queue_mutex);
		queue_mutex->unlock();
		//log4cplus_debug(kModuleName, "%s: The node is NULL", __func__);
		return NULL;
	}

	queue->front = queue->front->next;
	queue->size -= 1;
	//pthread_mutex_unlock(queue_mutex);
	queue_mutex->unlock();

	//log4cplus_debug(kModuleName, "%s: type=%s size=%d", __func__, type, queue->size);
	return element;
}

void FrameQueueProcess::ResetFreeQueue(FrameQueue *workQueue, FrameQueue *freeQueue) {
	if (workQueue == NULL) {
		//log4cplus_debug(kModuleName, "%s: The WorkQueue is NULL", __func__);
		return;
	}

	if (freeQueue == NULL) {
		//log4cplus_debug(kModuleName, "%s: The FreeQueue is NULL", __func__);
		return;
	}

	int workQueueSize = workQueue->size;
	if (workQueueSize > 0) {
		for (int i = 0; i < workQueueSize; i++) {
			FrameQueueNode *node = DeQueue(workQueue);
			free(node->data);
			node->data = NULL;
			EnQueue(freeQueue, node);
		}
	}
	//log4cplus_info(kModuleName, "%s: ResetFreeQueue : The work queue size is %d, free queue size is %d", __func__, workQueue->size, freeQueue->size);
}

void FrameQueueProcess::ClearFrameQueue(FrameQueue *queue) {
	while (queue->size) {
		FrameQueueNode *node = this->DeQueue(queue);
		this->FreeNode(node);
	}

	//log4cplus_info(kModuleName, "%s: Clear FrameQueueProcess queue", __func__);
}

void FrameQueueProcess::FreeNode(FrameQueueNode* node) {
	if (node != NULL) {
		free(node->data);
		free(node);
	}
}