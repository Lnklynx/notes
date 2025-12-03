from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_vector_store
from ..models import (
    VectorAddRequest,
    VectorUpdateRequest,
    VectorDeleteRequest,
    VectorSearchRequest,
    VectorQueryRequest,
    VectorQueryItem,
    VectorCollectionInfo,
    VectorCollectionListResponse,
)
from ..responses import success_response
from ...embedding.vector_store import VectorStore

router = APIRouter()


@router.post("/vector_store/add",
             summary="添加向量数据",
             description="向向量库添加新的向量数据")
def add_vectors(
        request: VectorAddRequest,
        vector_store: VectorStore = Depends(get_vector_store),
):
    """添加向量数据到向量库"""
    try:
        # 验证输入长度一致性
        ids_len = len(request.ids)
        if len(request.embeddings) != ids_len or len(request.documents) != ids_len:
            raise HTTPException(
                status_code=400,
                detail="ids、embeddings 和 documents 的长度必须一致"
            )
        if request.metadatas and len(request.metadatas) != ids_len:
            raise HTTPException(
                status_code=400,
                detail="metadatas 的长度必须与 ids 一致"
            )

        collection_name = request.collection_name or vector_store.DEFAULT_COLLECTION_NAME
        vector_store.add_documents(
            ids=request.ids,
            embeddings=request.embeddings,
            documents=request.documents,
            metadatas=request.metadatas,
            collection_name=collection_name,
        )
        return success_response(
            message=f"成功添加 {ids_len} 个向量到集合 '{collection_name}'"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加向量失败: {str(e)}")


@router.put("/vector_store/update",
            summary="更新向量数据",
            description="更新向量库中已存在的向量数据")
def update_vectors(
        request: VectorUpdateRequest,
        vector_store: VectorStore = Depends(get_vector_store),
):
    """更新向量库中的向量数据"""
    try:
        ids_len = len(request.ids)

        # 验证输入长度一致性
        if request.embeddings and len(request.embeddings) != ids_len:
            raise HTTPException(
                status_code=400,
                detail="embeddings 的长度必须与 ids 一致"
            )
        if request.documents and len(request.documents) != ids_len:
            raise HTTPException(
                status_code=400,
                detail="documents 的长度必须与 ids 一致"
            )
        if request.metadatas and len(request.metadatas) != ids_len:
            raise HTTPException(
                status_code=400,
                detail="metadatas 的长度必须与 ids 一致"
            )

        collection_name = request.collection_name or vector_store.DEFAULT_COLLECTION_NAME
        vector_store.update_documents(
            ids=request.ids,
            embeddings=request.embeddings,
            documents=request.documents,
            metadatas=request.metadatas,
            collection_name=collection_name,
        )
        return success_response(
            message=f"成功更新 {ids_len} 个向量"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新向量失败: {str(e)}")


@router.delete("/vector_store/delete",
               summary="删除向量数据",
               description="从向量库删除向量数据（支持按 ID 或 where 条件删除）")
def delete_vectors(
        request: VectorDeleteRequest,
        vector_store: VectorStore = Depends(get_vector_store),
):
    """删除向量库中的向量数据"""
    try:
        collection_name = request.collection_name or vector_store.DEFAULT_COLLECTION_NAME

        if request.ids:
            # 按 ID 列表删除
            vector_store.delete_by_ids(request.ids, collection_name=collection_name)
            return success_response(
                message=f"成功删除 {len(request.ids)} 个向量"
            )
        elif request.where:
            # 按 where 条件删除
            vector_store.delete_by_where(request.where, collection_name=collection_name)
            return success_response(
                message=f"成功删除满足条件的向量"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="必须提供 ids 或 where 条件之一"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除向量失败: {str(e)}")


@router.post("/vector_store/search",
             summary="向量检索",
             description="根据查询向量检索最相似的文档")
def search_vectors(
        request: VectorSearchRequest,
        vector_store: VectorStore = Depends(get_vector_store),
):
    """向量检索"""
    try:
        collection_name = request.collection_name or vector_store.DEFAULT_COLLECTION_NAME
        vector_store._ensure_collection(collection_name)

        results = vector_store.search(
            query_embedding=request.query_embedding,
            top_k=request.top_k,
            where=request.where,
        )

        # 格式化返回结果
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        items = []
        for i in range(len(ids)):
            items.append({
                "id": ids[i],
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "distance": distances[i] if i < len(distances) else None,
            })

        return success_response(data={"items": items})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量检索失败: {str(e)}")


@router.post("/vector_store/query",
             summary="元数据查询",
             description="根据元数据过滤条件查询向量库中的文档内容")
def query_vectors(request: VectorQueryRequest,
                  vector_store: VectorStore = Depends(get_vector_store), ):
    """按元数据查询向量库"""
    try:
        results = vector_store.search_by_metadata(
            where=request.where if request.where else None,
            top_k=request.limit,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]

        items = []
        for i in range(len(ids)):
            items.append(VectorQueryItem(
                id=ids[i],
                document=documents[i] if i < len(documents) else "",
                metadata=metadatas[i] if i < len(metadatas) else {},
            ))

        return success_response(data={"items": items})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"元数据查询失败: {str(e)}")


@router.get("/vector_store/collections",
            summary="列出所有集合",
            description="获取所有向量库集合的列表")
def list_collections(
        vector_store: VectorStore = Depends(get_vector_store),
):
    """列出所有向量库集合"""
    try:
        collections = vector_store.list_collections()
        return success_response(
            data=VectorCollectionListResponse(collections=collections)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"列出集合失败: {str(e)}")


@router.get("/vector_store/collections/{collection_name}",
            summary="获取集合信息",
            description="获取指定集合的详细信息")
def get_collection_info(
        collection_name: str,
        vector_store: VectorStore = Depends(get_vector_store),
):
    """获取集合信息"""
    try:
        info = vector_store.get_collection_info(collection_name)
        return success_response(data=VectorCollectionInfo(**info))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取集合信息失败: {str(e)}")


@router.delete("/vector_store/collections/{collection_name}",
               summary="删除集合",
               description="删除整个向量库集合（危险操作，会删除所有数据）")
def delete_collection(collection_name: str,
                      vector_store: VectorStore = Depends(get_vector_store), ):
    """删除整个向量库集合"""
    try:
        vector_store.delete_collection(collection_name)
        return success_response(
            message=f"成功删除集合 '{collection_name}'"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除集合失败: {str(e)}")
