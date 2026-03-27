import axios from 'axios';
import type { Post, Trace, ActionResponse } from '../types';

const API_BASE_URL = '/api/v1';

export const api = axios.create({
    baseURL: API_BASE_URL,
});

export const getPosts = async (limit: number = 20, offset: number = 0) => {
  const response = await api.get<Post[]>(`/posts`, { params: { limit, offset } });
  return response.data;
};

export const getTraces = async () => {
    const response = await api.get<Trace[]>('/traces');
    return response.data;
};

export const createPost = async (userId: number, content: string) => {
    const response = await api.post<ActionResponse>('/posts', { user_id: userId, content });
    return response.data;
};

export const likePost = async (userId: number, postId: number) => {
    const response = await api.post<ActionResponse>('/posts/like', { user_id: userId, post_id: postId });
    return response.data;
};

export const getPostDetail = async (postId: number, userId?: number) => {
    const response = await api.get<Post>(`/posts/${postId}`, { params: { user_id: userId } });
    return response.data;
};

export const createComment = async (userId: number, postId: number, content: string) => {
    const response = await api.post<ActionResponse>('/comments', { user_id: userId, post_id: postId, content });
    return response.data;
};

export const getTime = async () => {
    const response = await api.get<{ current_time: string; mode: string; step: number }>('/time');
    return response.data;
};
