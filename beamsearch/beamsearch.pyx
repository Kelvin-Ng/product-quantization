import numpy as np
cimport numpy as np

def encodePointsBeamSearch(int startPid, int pointsCount, np.ndarray[np.float32_t, ndim=2] pointCodebookProducts, float[:, :, :] codebooksProducts, float[:] codebooksNorms, int branch):
    cdef int M = codebooksProducts.shape[0]
    cdef int K = codebooksProducts.shape[1]
    cdef int[:] hashArray = np.array([13 ** i for i in range(M)], dtype=np.int32)
    pointsCount = min(pointsCount, pointCodebookProducts.shape[0] - startPid)
    cdef np.ndarray[np.int32_t, ndim=2] assigns = np.zeros((pointsCount, M), dtype=np.int32)
    cdef np.ndarray[np.float32_t, ndim=1] errors = np.zeros((pointsCount), dtype=np.float32)

    cdef np.ndarray[np.float32_t, ndim=1] distances
    cdef np.ndarray[np.int32_t, ndim=1] bestIdx
    cdef int[:] vocIds
    cdef int[:] wordIds
    cdef int[:, :] bestSums
    cdef int candidateIdx
    cdef float[:] bestSumScores
    cdef int m

    cdef np.ndarray[np.float32_t, ndim=1] candidatesScores
    cdef np.int8_t[:] globalHashTable
    cdef np.int64_t[:] bestIndices
    cdef int found
    cdef int currentBestIndex
    cdef int[:, :] newBestSums
    cdef float[:] newBestSumsScores

    cdef int bestIndex
    cdef int candidateId
    cdef int codebookId
    cdef int wordId
    cdef int hashIdx

    cdef int i

    cdef int pid
    for pid in range(startPid, startPid+pointsCount):
        distances = -pointCodebookProducts[pid,:] + codebooksNorms
        bestIdx = distances.argsort()[0:branch].astype(np.int32)
        vocIds = bestIdx // K
        wordIds = bestIdx % K
        bestSums = -1 * np.ones((branch, M), dtype=np.int32)
        for candidateIdx in range(branch):
            bestSums[candidateIdx,vocIds[candidateIdx]] = wordIds[candidateIdx]
        bestSumScores = distances[bestIdx]
        for m in range(1, M):
            candidatesScores = np.array([[bestSumScores[i]] * (M * K) for i in range(branch)], dtype=np.float32).flatten()
            candidatesScores += np.tile(distances, branch)
            globalHashTable = np.zeros(115249, dtype='int8')
            for candidateIdx in range(branch):
                for m in range(M):
                      if bestSums[candidateIdx,m] < 0:
                          continue
                      candidatesScores[candidateIdx*M*K:(candidateIdx+1)*M*K] += \
                          codebooksProducts[m, bestSums[candidateIdx,m], :]
                      candidatesScores[candidateIdx*M*K + m*K:candidateIdx*M*K+(m+1)*K] += 999999
            bestIndices = candidatesScores.argsort()
            found = 0
            currentBestIndex = 0
            newBestSums = -1 * np.ones((branch, M), dtype=np.int32)
            newBestSumsScores = -1 * np.ones((branch), dtype=np.float32)
            while found < branch:
                bestIndex = bestIndices[currentBestIndex]
                candidateId = bestIndex // (M * K)
                codebookId = (bestIndex % (M * K)) // K
                wordId = (bestIndex % (M * K)) % K
                bestSums[candidateId,codebookId] = wordId
                hashIdx = np.dot(bestSums[candidateId,:], hashArray) % 115249
                if globalHashTable[hashIdx] == 1:
                    bestSums[candidateId,codebookId] = -1
                    currentBestIndex += 1
                    continue
                else:
                    bestSums[candidateId,codebookId] = -1
                    globalHashTable[hashIdx] = 1
                    newBestSums[found,:] = bestSums[candidateId,:]
                    newBestSums[found,codebookId] = wordId
                    newBestSumsScores[found] = candidatesScores[bestIndex]
                    found += 1
                    currentBestIndex += 1
            bestSums = newBestSums.copy()
            bestSumScores = newBestSumsScores.copy()
        assigns[pid-startPid,:] = bestSums[0,:]
        errors[pid-startPid] = bestSumScores[0]
    return (assigns, errors)
