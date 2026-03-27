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
