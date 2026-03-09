export interface User {
    id: number;
    username: string;
    nickname: string;
    bio?: string;
    avatar_url?: string;
}

export interface Post {
    id: number;
    user_id: number;
    content: string;
    type: string;
    ref_id?: number;
    stats: {
        like_count: number;
        reply_count: number;
        share_count: number;
        retweet_count: number;
        quote_count: number;
    };
    created_at: string;
    author_nickname: string;
    author_type?: string;
    comments?: Comment[];
    is_liked?: boolean;
    is_reposted?: boolean;
}

export interface Comment {
    id: number;
    user_id: number;
    post_id: number;
    content: string;
    created_at: string;
    author: User; // The backend response flattens this to author_nickname/author_type but schema says comments have author_nickname.
    // Wait, the schema in schemas.py says CommentResponse has author_nickname, author_type.
    // My previous types.ts had author: User.
    // Let's check how I used it in PostDetail.tsx: comment.author.nickname
    // But the backend returns flat fields.
    // I need to align types with backend response.
    // Backend `get_post_detail` returns `c_dict` which has `author_nickname`.
    // It does NOT have a nested `author` object.
    // So my previous code in PostDetail.tsx using `comment.author.nickname` was probably wrong or I need to map it.
    // Let's check PostDetail.tsx again.
    // It uses `comment.author.nickname`.
    // So I need to fix PostDetail.tsx as well.
    author_nickname: string;
    author_type?: string;
    like_count: number;
    is_liked?: boolean;
}

export interface Trace {
    id: number;
    user_id: number;
    user_nickname: string;
    action_type: string;
    action_details?: Record<string, unknown>;
    created_at: string;
}

export interface ActionResponse {
    status: string;
    message: string;
    data?: unknown;
}
