"""
Search Module
-----------

Advanced search functionality for the RAG pipeline with comprehensive 
ranking and analysis capabilities.

Key Features:
- Vector search
- Hybrid ranking
- Context awareness
- Metadata filtering
- Result reranking
- Score normalization
- Performance analysis

Technical Details:
- Embedding handling
- Similarity calculation
- Score aggregation
- Filter processing
- Query enhancement
- Result caching
- Error handling

Dependencies:
- numpy>=1.24.0
- typing (standard library)
- logging (standard library)
- re (standard library)

Example Usage:
    # Initialize engine
    engine = SearchEngine()
    
    # Perform search with reranking
    results = engine.search(
        query_vec=query_embeddings,
        doc_vecs=document_embeddings,
        docs=documents,
        rerank_params={
            'page_position': 0.1,
            'metadata': 0.15
        }
    )
    
    # Analyze results
    analysis = engine.analyze_results(results)

Search Parameters:
- Vector similarity
- Token matching
- Page position
- Metadata matching
- Context relevance

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple, TypedDict, Any

from .tokenizer import default_tokenizer as rag_tokenizer
from .query import default_query_processor as query
import numpy as np
from src.utils.text_cleaner import TextCleaner
from src.utils.doc_store_conn import DocStoreConnection, MatchDenseExpr, FusionExpr, OrderByExpr
from src.rag.utils import rmSpace

PAGERANK_FLD = "pagerank_fea"
TAG_FLD = "tag_feas"
def index_name(uid): return f"ragflow_{uid}"



class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    class SearchResult:
        total: int
        ids: List[str]
        query_vector: Optional[List[float]] = None
        field: Optional[Dict] = None
        highlight: Optional[Dict] = None
        aggregation: Optional[Dict] = None
        keywords: Optional[List[str]] = None
        group_docs: Optional[List[List]] = None

    def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        qv, _ = emb_mdl.encode_queries(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    def get_filters(self, req):
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        # TODO(yzc): `available_int` is nullable however infinity doesn't support nullable columns.
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    def search(self, req, idx_names: Union[str, List[str]],
               kb_ids: List[str],
               emb_mdl=None,
               highlight=False,
               rank_feature: Optional[Dict] = None
               ):
        filters = self.get_filters(req)
        orderBy = OrderByExpr()

        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                       "doc_id", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                       "question_kwd", "question_tks",
                       "available_int", "content_with_weight", PAGERANK_FLD, TAG_FLD])
        kwds = set([])

        qst = req.get("question", "")
        q_vec = []
        if not qst:
            if req.get("sort"):
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.getTotal(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            highlightFields = ["content_ltks", "title_tks"] if highlight else []
            matchText, keywords = self.qryr.question(qst, min_match=0.3)
            if emb_mdl is None:
                matchExprs = [matchText]
                res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                matchDense = self.get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))
                q_vec = matchDense.embedding_data
                src.append(f"q_{len(q_vec)}_vec")

                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05, 0.95"})
                matchExprs = [matchText, matchDense, fusionExpr]

                res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))

                # If result is empty, try again with lower min_match
                if total == 0:
                    matchText, _ = self.qryr.question(qst, min_match=0.1)
                    filters.pop("doc_ids", None)
                    matchDense.extra_options["similarity"] = 0.17
                    res = self.dataStore.search(src, highlightFields, filters, [matchText, matchDense, fusionExpr],
                                                orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                    total = self.dataStore.getTotal(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        logging.debug(f"TOTAL: {total}")
        ids = self.dataStore.getChunkIds(res)
        keywords = list(kwds)
        highlight = self.dataStore.getHighlight(res, keywords, "content_with_weight")
        aggs = self.dataStore.getAggregation(res, "docnm_kwd")
        return self.SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec,
            aggregation=aggs,
            highlight=highlight,
            field=self.dataStore.getFields(res, src),
            keywords=keywords
        )

    @staticmethod
    def trans2floats(txt):
        return [float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0]*len(ans_v[0])
                logging.warning("The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i],
                                                                chunk_v,
                                                                rag_tokenizer.tokenize(
                                                                    self.qryr.rmWWW(pieces_[i])).split(),
                                                                chunks_tks,
                                                                tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" ##{c}$$"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea: Optional[Dict], search_res: 'Dealer.SearchResult') -> np.ndarray:
        ## For rank feature(tag_fea) scores.
        rank_fea = []
        pageranks = []
        for chunk_id in search_res.ids:
            pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
        pageranks = np.array(pageranks, dtype=float)

        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

        q_denor = np.sqrt(np.sum([s*s for t,s in query_rfea.items() if t != PAGERANK_FLD]))
        for i in search_res.ids:
            nor, denor = 0, 0
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor/np.sqrt(denor)/q_denor)
        return np.array(rank_fea)*10. + pageranks

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: Optional[Dict] = None
               ):
        _, keywords = self.qryr.question(query)
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: Optional[Dict] = None):
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = rerank_mdl.similarity(query, [" ".join(tks).strip() for tks in ins_tw])
        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * (np.array(tksim)+rank_fea) + vtweight * vtsim, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(),
                                           rag_tokenizer.tokenize(inst).split())

    def retrieval(self, question, embd_mdl, tenant_ids, kb_ids, page, page_size, similarity_threshold=0.2,
                  vector_similarity_weight=0.3, top=1024, doc_ids=None, aggs=True,
                  rerank_mdl=None, highlight=False,
                  rank_feature: Optional[Dict] = {PAGERANK_FLD: 10}):
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks

        RERANK_PAGE_LIMIT = 3
        req = {"kb_ids": kb_ids, "doc_ids": doc_ids, "size": max(page_size * RERANK_PAGE_LIMIT, 128),
               "question": question, "vector": True, "topk": top,
               "similarity": similarity_threshold,
               "available_int": 1}

        if page > RERANK_PAGE_LIMIT:
            req["page"] = page
            req["size"] = page_size

        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        sres = self.search(req, [index_name(tid) for tid in tenant_ids],
                           kb_ids, embd_mdl, highlight, rank_feature=rank_feature)
        ranks["total"] = sres.total

        if page <= RERANK_PAGE_LIMIT:
            if rerank_mdl and sres.total > 0:
                sim, tsim, vsim = self.rerank_by_model(rerank_mdl,
                                                       sres, question, 1 - vector_similarity_weight,
                                                       vector_similarity_weight,
                                                       rank_feature=rank_feature)
            else:
                sim, tsim, vsim = self.rerank(
                    sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
                    rank_feature=rank_feature)
            idx = np.argsort(sim * -1)[(page - 1) * page_size:page * page_size]
        else:
            sim = tsim = vsim = [1] * len(sres.ids)
            idx = list(range(len(sres.ids)))

        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim
        for i in idx:
            if sim[i] < similarity_threshold:
                break
            if len(ranks["chunks"]) >= page_size:
                if aggs:
                    continue
                break
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk["docnm_kwd"]
            did = chunk["doc_id"]
            position_int = chunk.get("position_int", [])
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": chunk["doc_id"],
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim[i],
                "vector_similarity": vsim[i],
                "term_similarity": tsim[i],
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
            }
            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = rmSpace(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)
            if dnm not in ranks["doc_aggs"]:
                ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
            ranks["doc_aggs"][dnm]["count"] += 1
        ranks["doc_aggs"] = [{"doc_name": k,
                              "doc_id": v["doc_id"],
                              "count": v["count"]} for k,
                                                       v in sorted(ranks["doc_aggs"].items(),
                                                                   key=lambda x: x[1]["count"] * -1)]
        ranks["chunks"] = ranks["chunks"][:page_size]

        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(self, doc_id: str, tenant_id: str,
                   kb_ids: List[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"]):
        condition = {"doc_id": doc_id}
        res = []
        bs = 128
        for p in range(offset, max_count, bs):
            es_res = self.dataStore.search(fields, [], condition, [], OrderByExpr(), p, bs, index_name(tenant_id),
                                           kb_ids)
            dict_chunks = self.dataStore.getFields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id
            if dict_chunks:
                res.extend(dict_chunks.values())
            if len(dict_chunks.values()) < bs:
                break
        return res

    def all_tags(self, tenant_id: str, kb_ids: List[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.getAggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: List[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.getAggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: List[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a: c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: Union[str, List[str]], kb_ids: List[str], all_tags, topn_tags=3, S=1000):
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        return {a: max(1, c) for a, c in tag_fea}

class SearchEngine:
    """Search engine for RAG pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.qryr = query.FulltextQueryer()
        self.text_cleaner = TextCleaner()
    
    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        return dot_product / (norm_v1 * norm_v2)

    def semantic_similarity(
        self,
        text1: str,
        text2: str,
        embeddings_dict: Dict[str, np.ndarray]
    ) -> float:
        """Calculate semantic similarity between texts using cached embeddings."""
        if text1 in embeddings_dict and text2 in embeddings_dict:
            return self.cosine_similarity(
                embeddings_dict[text1],
                embeddings_dict[text2]
            )
        return 0.0
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess search query."""
        query = self.text_cleaner.clean_text(query)
        query = self.text_cleaner.normalize_whitespace(query)
        tokens = self.qryr.tokenize(query)
        filtered_tokens = self.qryr.filter_tokens(
            tokens,
            exclude_stops=True
        )
        return self.qryr.decode(filtered_tokens)
    
    def rerank_by_page_position(
        self,
        results: List[Dict],
        alpha: float = 0.1
    ) -> List[Dict]:
        """Rerank results considering page position."""
        for result in results:
            page_position = result['document'].get('page_number', 0)
            position_score = np.exp(-alpha * page_position)
            result['score'] = result['score'] * (1 + position_score)
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def rerank_by_context(
        self,
        results: List[Dict],
        query: str,
        embeddings_dict: Dict[str, np.ndarray],
        beta: float = 0.2
    ) -> List[Dict]:
        """Rerank results considering surrounding context."""
        for result in results:
            context = result['document'].get('context', '')
            context_score = self.semantic_similarity(
                query,
                context,
                embeddings_dict
            )
            result['score'] = result['score'] * (1 + beta * context_score)
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def rerank_by_metadata(
        self,
        results: List[Dict],
        target_metadata: Dict[str, Union[str, float]],
        gamma: float = 0.15
    ) -> List[Dict]:
        """Rerank results based on metadata matching."""
        for result in results:
            metadata_score = 0.0
            doc_metadata = result['document'].get('metadata', {})
            
            for key, target_value in target_metadata.items():
                if key in doc_metadata:
                    if isinstance(target_value, (int, float)):
                        # Numerical comparison
                        diff = abs(float(doc_metadata[key]) - float(target_value))
                        metadata_score += 1.0 / (1.0 + diff)
                    else:
                        # String comparison
                        metadata_score += float(doc_metadata[key] == target_value)
            
            avg_metadata_score = metadata_score / len(target_metadata)
            result['score'] = result['score'] * (1 + gamma * avg_metadata_score)
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def search(
        self,
        query_embedding: np.ndarray,
        document_embeddings: List[np.ndarray],
        documents: List[Dict],
        query: Optional[str] = None,
        embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
        metadata_weights: Optional[Dict[str, Union[str, float]]] = None,
        top_k: int = 3,
        threshold: float = 0.0,
        rerank_params: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Enhanced search with multiple reranking options.
        
        Args:
            query_embedding: Query vector
            document_embeddings: List of document vectors
            documents: List of document dictionaries with metadata
            query: Original query text for context reranking
            embeddings_dict: Cache of text embeddings
            metadata_weights: Target metadata for reranking
            top_k: Number of results to return
            threshold: Minimum similarity score
            rerank_params: Dictionary of reranking parameters
                {
                    'page_position': float,  # alpha
                    'context': float,        # beta
                    'metadata': float        # gamma
                }
        """
        # Initial similarity search
        similarities = [
            self.cosine_similarity(query_embedding, doc_embedding)
            for doc_embedding in document_embeddings
        ]
        
        # Filter by threshold
        results = [
            {
                'document': documents[idx],
                'score': score,
                'index': idx
            }
            for idx, score in enumerate(similarities)
            if score >= threshold
        ]
        
        # Apply reranking if parameters provided
        if rerank_params:
            if rerank_params.get('page_position'):
                results = self.rerank_by_page_position(
                    results,
                    alpha=rerank_params['page_position']
                )
            
            if rerank_params.get('context') and query and embeddings_dict:
                results = self.rerank_by_context(
                    results,
                    query,
                    embeddings_dict,
                    beta=rerank_params['context']
                )
            
            if rerank_params.get('metadata') and metadata_weights:
                results = self.rerank_by_metadata(
                    results,
                    metadata_weights,
                    gamma=rerank_params['metadata']
                )
        
        return results[:top_k]
    
    def batch_search(
        self,
        query_embeddings: List[np.ndarray],
        document_embeddings: List[np.ndarray],
        documents: List[Dict],
        queries: Optional[List[str]] = None,
        embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
        metadata_weights: Optional[Dict[str, Union[str, float]]] = None,
        top_k: int = 3,
        threshold: float = 0.0,
        rerank_params: Optional[Dict] = None
    ) -> List[List[Dict]]:
        """Perform batch search with reranking."""
        return [
            self.search(
                query_embedding,
                document_embeddings,
                documents,
                query=queries[i] if queries else None,
                embeddings_dict=embeddings_dict,
                metadata_weights=metadata_weights,
                top_k=top_k,
                threshold=threshold,
                rerank_params=rerank_params
            )
            for i, query_embedding in enumerate(query_embeddings)
        ]
    
    def analyze_results(
        self,
        results: List[Dict],
        min_score: float = 0.0
    ) -> Dict:
        """Analyze search results."""
        return {
            'total_results': len(results),
            'average_score': np.mean([r['score'] for r in results]),
            'max_score': max(r['score'] for r in results),
            'min_score': min(r['score'] for r in results),
            'above_threshold': len([r for r in results if r['score'] >= min_score]),
            'metadata_coverage': self._analyze_metadata_coverage(results)
        }
    
    def _analyze_metadata_coverage(self, results: List[Dict]) -> Dict:
        """Analyze metadata coverage in results."""
        metadata_fields = set()
        field_counts = {}
        
        for result in results:
            metadata = result['document'].get('metadata', {})
            for field in metadata:
                metadata_fields.add(field)
                field_counts[field] = field_counts.get(field, 0) + 1
        
        return {
            'total_fields': len(metadata_fields),
            'field_coverage': {
                field: count/len(results)
                for field, count in field_counts.items()
            }
        }

# Example usage
if __name__ == "__main__":
    engine = SearchEngine()
    
    # Example data
    query_vec = np.random.rand(768)
    doc_vecs = [np.random.rand(768) for _ in range(10)]
    docs = [{
        'id': i,
        'text': f'Document {i}',
        'page_number': i,
        'metadata': {
            'subject': 'physics',
            'grade_level': 10
        }
    } for i in range(10)]
    
    # Search with reranking
    results = engine.search(
        query_vec,
        doc_vecs,
        docs,
        rerank_params={
            'page_position': 0.1,
            'metadata': 0.15
        },
        metadata_weights={
            'subject': 'physics',
            'grade_level': 10
        }
    )
    
    # Analyze results
    analysis = engine.analyze_results(results)
    print("Search Analysis:", analysis)
