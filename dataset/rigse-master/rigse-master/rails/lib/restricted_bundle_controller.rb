module RestrictedBundleController

  # restrict access to admins
  def self.included(clazz)
    include RestrictedController
    clazz.class_eval {
      # TODO: do console loggers use :bundle format too?
      # in the meantime, I have added the :except => rules to be sure.
      before_action :admin_only, :except => [:new, :create]
    protected
      def admin_only
        unless (current_visitor != nil && current_visitor.has_role?('admin')) || request.format == :bundle
          raise Pundit::NotAuthorizedError
        end
      end
    }
  end
end
