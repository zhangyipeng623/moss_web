import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getPostDetail } from '../lib/api';
import type { Post, Comment } from '../types';
import { PostCard } from '../components/PostCard';
import { ChevronLeft, Loader2, Heart } from 'lucide-react';
import { formatDistance } from 'date-fns';

interface PostDetailProps {
    systemTime: number;
}

export function PostDetail({ systemTime }: PostDetailProps) {
    const { id } = useParams<{ id: string }>();
    const [post, setPost] = useState<Post | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!id) {
            setError('帖子 ID 无效');
            setIsLoading(false);
            return;
        }

        setIsLoading(true);
        setError(null);

        getPostDetail(parseInt(id, 10))
            .then(data => {
                setPost(data);
            })
            .catch(err => {
                console.error(err);
                setPost(null);
                setError('帖子加载失败');
            })
            .finally(() => {
                setIsLoading(false);
            });
    }, [id]);

    if (isLoading) {
        return <div className="flex justify-center p-10"><Loader2 className="animate-spin" /></div>;
    }

    if (error || !post) {
        return (
            <div className="min-h-screen bg-slate-50 pb-20">
                <div className="sticky top-0 bg-white/80 backdrop-blur border-b border-slate-200 p-4 flex items-center gap-4 z-10 shadow-sm">
                    <Link to="/" className="p-2 hover:bg-slate-100 rounded-full transition-colors text-slate-600">
                        <ChevronLeft />
                    </Link>
                    <h1 className="text-xl font-bold text-slate-800">Post</h1>
                </div>
                <div className="max-w-2xl mx-auto p-8 text-center text-slate-500">
                    {error || '帖子不存在'}
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-slate-50 pb-20">
            {/* Header */}
            <div className="sticky top-0 bg-white/80 backdrop-blur border-b border-slate-200 p-4 flex items-center gap-4 z-10 shadow-sm">
                <Link to="/" className="p-2 hover:bg-slate-100 rounded-full transition-colors text-slate-600">
                    <ChevronLeft />
                </Link>
                <h1 className="text-xl font-bold text-slate-800">Post</h1>
            </div>

            <div className="max-w-2xl mx-auto p-4">
                <PostCard post={post} systemTime={systemTime} expanded={true} />

                <div className="mt-6 border-t border-slate-200 pt-6">
                    <h3 className="text-lg font-bold mb-4 text-slate-800">Comments</h3>

                    <div className="space-y-4">
                        {post.comments && post.comments.length > 0 ? (
                            post.comments.map((comment: Comment) => (
                                <div key={comment.id} className="flex gap-3 p-3 rounded-lg hover:bg-white/50 transition-colors border border-transparent hover:border-slate-200">
                                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center text-xs font-bold text-white shadow-sm">
                                        {comment.author_nickname?.charAt(0)}
                                    </div>
                                    <div className="flex-1">
                                        <div className="flex items-center justify-between mb-1">
                                            <div className="flex items-center gap-2">
                                                <span className="font-bold text-sm text-slate-800">{comment.author_nickname}</span>
                                                <span className="text-slate-500 text-xs">@{comment.author_nickname}</span>
                                                <span className="text-slate-400 text-xs">
                                                    • {formatDistance(
                                                        new Date(comment.created_at),
                                                        new Date(systemTime),
                                                        { addSuffix: true }
                                                    )}
                                                </span>
                                            </div>
                                            <div className="flex items-center gap-1 text-slate-400 text-xs">
                                                <Heart size={12} className={comment.is_liked ? "fill-rose-500 text-rose-500" : ""} />
                                                <span>{comment.like_count}</span>
                                            </div>
                                        </div>
                                        <p className="text-slate-700 text-sm leading-relaxed">{comment.content}</p>
                                    </div>
                                </div>
                            ))
                        ) : (
                            <div className="text-center text-slate-400 py-4">No comments yet</div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
