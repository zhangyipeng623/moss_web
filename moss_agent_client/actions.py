from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    platform_method: str
    arg_names: tuple[str, ...] = ()
    result_mode: str = "action"


ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        name="create_post",
        description="发布一条新的帖子。",
        platform_method="create_post",
        arg_names=("content",),
    ),
    ActionSpec(
        name="create_comment",
        description="对指定帖子发布评论。",
        platform_method="create_comment",
        arg_names=("post_id", "content"),
    ),
    ActionSpec(
        name="like_post",
        description="给指定帖子点赞。",
        platform_method="like_post",
        arg_names=("post_id",),
    ),
    ActionSpec(
        name="like_comment",
        description="给指定评论点赞。",
        platform_method="like_comment",
        arg_names=("comment_id",),
    ),
    ActionSpec(
        name="repost",
        description="转发帖子，不附带新的内容。",
        platform_method="repost",
        arg_names=("post_id",),
    ),
    ActionSpec(
        name="quote",
        description="引用帖子并附带新的内容。",
        platform_method="quote",
        arg_names=("post_id", "content"),
    ),
    ActionSpec(
        name="do_nothing",
        description="本轮不执行任何动作。",
        platform_method="do_nothing",
    ),
    ActionSpec(
        name="get_post",
        description="查看指定帖子的详情与评论。",
        platform_method="get_post",
        arg_names=("post_id",),
        result_mode="post_detail",
    ),
)


ACTION_SPEC_BY_NAME = {spec.name: spec for spec in ACTION_SPECS}
