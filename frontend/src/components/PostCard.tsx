import { Link } from 'react-router-dom';
import { formatDistance } from 'date-fns';
import { Heart, MessageCircle, Repeat, MessageSquareQuote } from 'lucide-react';
import { clsx } from 'clsx';
import type { Post } from '../types';

interface PostCardProps {
    post: Post;
    systemTime?: number;
    expanded?: boolean;
}

export function PostCard({ post, systemTime, expanded = false }: PostCardProps) {
    const displayTime = formatDistance(
        new Date(post.created_at),
        systemTime ? new Date(systemTime) : new Date(),
        { addSuffix: true }
    );

    return (
        <div className="bg-white/80 backdrop-blur border border-slate-200 rounded-xl p-4 hover:bg-white/90 transition-colors cursor-pointer group shadow-sm hover:shadow-md">
            <Link to={`/post/${post.id}`} className="block">
                <div className="flex gap-3">
                    {/* Avatar Placeholder */}
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex-shrink-0 flex items-center justify-center text-white font-bold shadow-sm">
                        {post.author_nickname?.charAt(0) || '?'}
                    </div>

                    <div className="flex-1 min-w-0">
                        {/* Header */}
                        <div className="flex items-center gap-2 mb-1">
                            <span className="font-bold text-slate-900 truncate">{post.author_nickname || 'Unknown'}</span>
                            <span className="text-slate-300 text-xs ">{post.author_type || 'Unknown'}</span>
                            <span className="text-slate-400 text-xs">•</span>
                            <span className="text-slate-500 text-xs hover:underline">
                                {displayTime}
                            </span>
                        </div>

                        {/* Content */}
                        <p className={clsx(
                            "text-slate-800 text-sm leading-relaxed whitespace-pre-wrap mb-3 font-medium",
                            !expanded && "line-clamp-3 overflow-hidden"
                        )}>
                            {post.content}
                        </p>

                        {/* Actions */}
                        <div className="flex items-center justify-between text-slate-500 max-w-md">
                            <div className="flex items-center gap-1.5 text-slate-400">
                                <div className="p-1.5">
                                    <MessageCircle size={16} />
                                </div>
                                <span className="text-xs font-medium">{post.stats?.reply_count > 0 && post.stats.reply_count}</span>
                            </div>

                            <div className={clsx(
                                "flex items-center gap-1.5",
                                post.is_reposted ? "text-green-500" : "text-slate-400"
                            )}>
                                <div className="p-1.5">
                                    <Repeat size={16} className={clsx(post.is_reposted && "text-green-500")} />
                                </div>
                                <span className="text-xs font-medium">{post.stats?.retweet_count > 0 && post.stats.retweet_count}</span>
                            </div>

                            <div className="flex items-center gap-1.5 text-slate-400">
                                <div className="p-1.5">
                                    <MessageSquareQuote size={16} />
                                </div>
                                <span className="text-xs font-medium">{post.stats?.quote_count > 0 && post.stats.quote_count}</span>
                            </div>

                            <div className={clsx(
                                "flex items-center gap-1.5",
                                post.is_liked ? "text-rose-500" : "text-slate-400"
                            )}>
                                <div className="p-1.5">
                                    <Heart size={16} className={clsx(post.is_liked && "fill-current")} />
                                </div>
                                <span className="text-xs font-medium">{post.stats?.like_count > 0 && post.stats.like_count}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </Link>
        </div>
    );
}
